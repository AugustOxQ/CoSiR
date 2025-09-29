from typing import Optional, Dict, Any
import wandb


class SimpleWandbLogger:
    def __init__(self, run: Optional[wandb.sdk.wandb_run.Run] = None) -> None:
        self._run = run

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        try:
            wandb.finish()
        except Exception:
            pass
