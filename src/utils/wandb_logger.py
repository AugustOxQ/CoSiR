from typing import Optional, Dict, Any, Union
import wandb


class WandbLogger:
    """
    Wandb logger with epoch-based x-axis support, prefix namespacing,
    and automatic nested-dict flattening.

    Phase-specific methods (log_train / log_test / log_eval) automatically
    attach the corresponding epoch counter so wandb can use it as x-axis.
    Call wandb.define_metric() is done once at construction.
    """

    def __init__(self, run: Optional[wandb.sdk.wandb_run.Run] = None) -> None:
        self._run = run
        self._setup_epoch_axes()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _setup_epoch_axes(self) -> None:
        """Wire up epoch counters as x-axes for each logging phase."""
        wandb.define_metric("train_epoch")
        wandb.define_metric("eval_epoch")
        wandb.define_metric("test_epoch")
        # train/ = epoch summaries; train_loss/, train_monitor/, train_details/ = batch-level
        for _s in ("", "_loss", "_monitor", "_details"):
            wandb.define_metric(f"train{_s}/*", step_metric="train_epoch")
        wandb.define_metric("vis/*", step_metric="test_epoch")
        # Test groups — each gets its own wandb section (test_oracle/*, etc.)
        for _g in ("oracle", "raw", "diff", "oracle_img", "oracle_imgtxt"):
            wandb.define_metric(f"test_{_g}/*", step_metric="test_epoch")
        # Eval groups — each gets its own wandb section (eval_retrieval_gain/*, etc.)
        for _g in (
            "magnitude_effect",
            "condition_distance_correlation",
            "retrieval_gain",
            "diversity",
            "best_condition_upper_bound",
            "space_quality",
        ):
            wandb.define_metric(f"eval_{_g}/*", step_metric="eval_epoch")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _flatten(
        self, d: Dict[str, Any], prefix: str = "", sep: str = "/"
    ) -> Dict[str, Any]:
        """Recursively flatten a nested dict, joining keys with sep."""
        result: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten(v, key, sep))
            else:
                result[key] = v
        return result

    @staticmethod
    def _strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    # ------------------------------------------------------------------ #
    # Core log                                                             #
    # ------------------------------------------------------------------ #

    def log(
        self,
        metrics: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Generic log. Flattens nested dicts, optionally prepends prefix,
        strips None values, then calls wandb.log().

        step is accepted for API compatibility but intentionally not forwarded
        to wandb.log() — mixing explicit-step and stepless calls causes
        monotonicity warnings. Use train/step as a logged data field instead.
        """
        flat = self._flatten(metrics)
        if prefix:
            flat = {f"{prefix}/{k}": v for k, v in flat.items()}
        flat = self._strip_none(flat)
        if not flat:
            return
        wandb.log(flat)

    # ------------------------------------------------------------------ #
    # Phase-specific helpers                                               #
    # ------------------------------------------------------------------ #

    def log_train(
        self,
        metrics: Dict[str, Any],
        epoch: int,
        step: Optional[int] = None,
        section: Optional[str] = None,
    ) -> None:
        """
        Log training-phase metrics.

        section=None  → keys under train/*       (epoch-level summaries)
        section="loss"    → train_loss/*          (batch loss components)
        section="monitor" → train_monitor/*       (diagnostic metrics)
        section="details" → train_details/*       (step/batch counters)

        train_epoch is appended so wandb can use it as x-axis.
        step is accepted for API compatibility but not forwarded to wandb.log().
        """
        ns = f"train_{section}" if section else "train"
        data = self._flatten(metrics, prefix=ns)
        data["train_epoch"] = epoch
        data = self._strip_none(data)
        wandb.log(data)

    def log_test(
        self,
        metrics: Union[Dict[str, Any], Any],  # also accepts MetricResult
        epoch: int,
    ) -> None:
        """
        Log test-phase metrics.

        Accepts a plain dict or a MetricResult (anything with .metrics).
        All keys are expected to already carry the full test/* prefix
        (as produced by TestEvaluator). test_epoch is appended.
        """
        if hasattr(metrics, "metrics"):
            metrics = metrics.metrics
        data = dict(metrics)
        data["test_epoch"] = epoch
        data = self._strip_none(data)
        wandb.log(data)

    def log_eval(
        self,
        metrics: Dict[str, Any],
        epoch: int,
        prefix: Optional[str] = None,
    ) -> None:
        """
        Log eval-phase metrics (e.g. from CoSiRAutomaticEvaluator).

        Expects keys in "group/subkey" format. Each group is promoted to its
        own top-level wandb section as eval_<group>/<subkey>, so the UI shows
        one panel per group rather than one crowded "eval" panel.
        eval_epoch is appended so wandb can use it as x-axis.
        """
        data: Dict[str, Any] = {}
        for k, v in metrics.items():
            if "/" in k:
                group, rest = k.split("/", 1)
                top = f"eval_{prefix}_{group}" if prefix else f"eval_{group}"
                data[f"{top}/{rest}"] = v
            else:
                data[f"eval_{prefix}/{k}" if prefix else f"eval/{k}"] = v
        data["eval_epoch"] = epoch
        data = self._strip_none(data)
        wandb.log(data)

    # ------------------------------------------------------------------ #
    # Backward-compat                                                      #
    # ------------------------------------------------------------------ #

    def log_metrics(
        self, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        """Backward-compatible alias for log()."""
        self.log(metrics, step=step)

    def finish(self) -> None:
        try:
            wandb.finish()
        except Exception:
            pass
