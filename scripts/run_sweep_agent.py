"""
wandb sweep agent bridge for CoSiR lr / lr_label grid search.

Usage
-----
  # 1. Create the sweep (once, from repo root):
  #    wandb sweep scripts/sweep_config.yaml
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

# Static overrides that mirror run_local.sh (minus the swept lr / lr_label).
# wandb.enabled=false because the sweep agent already owns the wandb run.
_STATIC_OVERRIDES = [
    "dataset=impressions",
    "model=clip_base",
    "eval.evaluation_interval=100",
    "eval.oracle_aggregation=max",
    "loss.lambda_collapse=0.1",
    "loss.lambda_contrastive=1",
    "loss.lambda_laplacian=30",
    "loss.lambda_mixup=1",
    "loss.lambda_delta=0.1",
    "model.num_layers=6",
    "model.embedding_dim=16",
    "train.epochs=1000",
    "train.normalize=false",
    "train.imgtxt_factor=1",
    "train.initialization_strategy=imgtxt",
    "wandb.enabled=false",
]


def _run():
    with wandb.init() as run:
        lr = wandb.config.lr
        lr_label = wandb.config.lr_label

        overrides = _STATIC_OVERRIDES + [
            f"optimizer.lr={lr}",
            f"optimizer.lr_label={lr_label}",
            # Use the wandb run ID so experiment dirs are unique across parallel agents.
            f"experiment.name=sweep_{run.id}",
        ]

        with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config.yaml", overrides=overrides)

        print("Sweep run config:\n" + OmegaConf.to_yaml(cfg))
        # Tag each run with its lr values so they're readable in the runs table.
        run.config.update({"lr": lr, "lr_label": lr_label})
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
