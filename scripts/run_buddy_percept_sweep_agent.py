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

# Lives for this Python process only. With the sweep YAML's `program:`
# target, `wandb agent` starts a fresh process per trial, so this cache
# never carries over between trials: every trial re-extracts GoEmotions
# features (~40% of trial compute in sweep 40i43gt5). To actually reuse
# it, drive trials in-process with `wandb.agent(sweep_id, function=main)`
# or cache the extracted features on disk.
_CACHE = FixedInputCache()


def main() -> None:
    run = wandb.init()
    config = resolve_trial_config(dict(wandb.config))
    fixed_inputs = _CACHE.get(content_pca_dim=config.content_pca_dim,
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
