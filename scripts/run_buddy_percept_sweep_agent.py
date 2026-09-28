"""W&B agent entrypoint for the buddy-percept comprehensive sweep.
Usage (per spec §8): after `wandb sweep scripts/sweep_config_buddy_percept.yaml`,
run `wandb agent <sweep_id>` with this as the sweep's `program:` target.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

from scripts.buddy_percept_sweep.cache import FixedInputCache
from scripts.buddy_percept_sweep.config import resolve_trial_config
from scripts.buddy_percept_sweep.pipeline import run_trial
from scripts.buddy_percept_sweep.real_data import load_real_raw_inputs

_CACHE = FixedInputCache()  # lives for the whole agent process (spec §3.1)


def main() -> None:
    run = wandb.init()
    config = resolve_trial_config(dict(wandb.config))
    fixed_inputs = _CACHE.get(content_pca_dim=getattr(wandb.config, "content_pca_dim", 50),
                              raw_loader=load_real_raw_inputs)

    def log_checkpoint(epoch: int, proxy: float) -> None:
        wandb.log({"checkpoint_proxy": proxy, "epoch": epoch})

    result = run_trial(config, fixed_inputs, log_checkpoint=log_checkpoint)
    wandb.log({
        "objective": result.objective,
        "stage1_emotion_ami": result.emotion_ami,
        "stage1_genre_ami": result.genre_ami,
        "stage2_macro_auc": result.stage2_macro_auc,
        "n_topics_after_merge": result.n_topics_after_merge,
        "stage1_seconds": result.stage1_seconds,
        "stage2_seconds": result.stage2_seconds,
    })
    run.finish()


if __name__ == "__main__":
    main()
