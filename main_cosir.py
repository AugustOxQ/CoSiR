import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from typing import Any, Dict, cast
from accelerate import Accelerator


from src.utils import setup_seed, SimpleWandbLogger
from src.hook import train_cosir


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    setup_seed(cfg.seed)

    wandb_logger = None
    if cfg.get("wandb") and cfg.wandb.enabled:
        if getattr(cfg.wandb, "mode", "") == "disabled":
            os.environ["WANDB_MODE"] = "disabled"
        elif getattr(cfg.wandb, "mode", "") == "offline":
            os.environ["WANDB_MODE"] = "offline"
        run_config = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=getattr(cfg.wandb, "project", None),
            entity=getattr(cfg.wandb, "entity", None),
            name=(cfg.wandb.name if getattr(cfg.wandb, "name", "") else None),
            config=run_config,
            tags=(list(cfg.wandb.tags) if getattr(cfg.wandb, "tags", None) else None),
            notes=(cfg.wandb.notes if getattr(cfg.wandb, "notes", None) else None),
        )

        wandb_logger = SimpleWandbLogger()

    results = train_cosir(cfg, wandb_logger)

    if cfg.get("wandb") and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
