"""
wandb sweep agent bridge for CoSiR grid search.

Supports v1 (lr/lr_label), v2 (+ scheduler_type/em_interval) and v3
(embedding_dim/lr/lr_label/buddies.alpha) sweep configs with the same agent;
parameters not present in a given sweep fall back to the config defaults.

Usage
-----
  # 1. Create the sweep (once, from repo root):
  #    wandb sweep scripts/sweep_config_v3.yaml
  #    → prints: "Created sweep: <entity>/<project>/<sweep_id>"

  # 2. Launch one agent per GPU:
  #    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_agent.py --sweep_id <sweep_id>
  #    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep_agent.py --sweep_id <sweep_id>
  #
  #    <sweep_id> can be the short hash or the full entity/project/hash form.
  #    Use --count N to limit how many runs this agent picks up.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.utils import WandbLogger
from src.hook import train_cosir


_CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

# W&B group for the sweep runs (overridable via env). Set on wandb.init() so all
# runs from this sweep are grouped together in the UI.
_WANDB_GROUP = os.environ.get("WANDB_RUN_GROUP", "condition buddy setting")

# Static overrides shared by every run in the conditional-buddies (v3) sweep.
# Mirrors the static portion of the multirun command; everything not listed here
# (loss weights, scheduler, etc.) intentionally falls back to the config defaults
# in configs/train/default.yaml. wandb.enabled=false because the sweep agent
# already owns the wandb run.
_STATIC_OVERRIDES = [
    "dataset=impressions",
    "model=clip_base",
    "eval.evaluation_interval=100",
    "eval.oracle_aggregation=max",
    "model.num_layers=6",
    "train.epochs=1000",
    'experiment.results_dir="res/CoSiR_Experiment_buddy2/impressions"',
    "wandb.enabled=false",
]


def _run():
    with wandb.init(group=_WANDB_GROUP) as run:
        lr = wandb.config.lr
        lr_label = wandb.config.lr_label
        # embedding_dim / alpha are swept in v3; scheduler_type / em_interval are
        # only present in v2 sweeps. Fall back to the config defaults when a given
        # sweep doesn't sweep that parameter, so the same agent runs v1/v2/v3.
        embedding_dim = getattr(wandb.config, "embedding_dim", None)
        alpha = getattr(wandb.config, "alpha", None)
        scheduler_type = getattr(wandb.config, "scheduler_type", None)
        em_interval = getattr(wandb.config, "em_interval", None)

        overrides = _STATIC_OVERRIDES + [
            f"optimizer.lr={lr}",
            f"optimizer.lr_label={lr_label}",
            # Use the wandb run ID so experiment dirs are unique across parallel agents.
            f"experiment.name=sweep_{run.id}",
        ]
        if embedding_dim is not None:
            overrides.append(f"model.embedding_dim={embedding_dim}")
        if alpha is not None:
            overrides.append(f"train.buddies.alpha={alpha}")
        if scheduler_type is not None:
            overrides.append(f"scheduler.type={scheduler_type}")
        if em_interval is not None:
            overrides.append(f"train.em_interval={em_interval}")

        with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config.yaml", overrides=overrides)

        print("Sweep run config:\n" + OmegaConf.to_yaml(cfg))
        # Tag each run with all swept values so they appear in the runs table.
        run.config.update({
            "lr": lr,
            "lr_label": lr_label,
            **({"embedding_dim": embedding_dim} if embedding_dim is not None else {}),
            **({"alpha": alpha} if alpha is not None else {}),
            **({"scheduler_type": scheduler_type} if scheduler_type is not None else {}),
            **({"em_interval": em_interval} if em_interval is not None else {}),
        })
        logger = WandbLogger()
        train_cosir(cfg, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a wandb sweep agent for CoSiR.")
    parser.add_argument(
        "--sweep_id",
        required=True,
        help="Sweep ID returned by 'wandb sweep', e.g. entity/project/abc123 or just abc123",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Max number of runs for this agent to execute (default: all remaining).",
    )
    args = parser.parse_args()

    wandb.agent(args.sweep_id, function=_run, count=args.count)
