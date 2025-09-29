"""Unified evaluation interface for easy integration."""

from typing import Optional, List, Union, Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader

from .config import EvaluationConfig, MetricResult
from .evaluators import TrainEvaluator, TestEvaluator


class EvaluationManager:
    """Unified interface for all evaluation tasks."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluation manager with configuration."""
        self.config = config or EvaluationConfig()
        self.train_evaluator = TrainEvaluator(self.config)
        self.test_evaluator = TestEvaluator(self.config)

    def evaluate_train(
        self,
        model,
        feature_manager,
        embedding_manager,
        dataloader: DataLoader,
        device: Optional[str] = None,
        epoch: int = 0,
        k_vals: Optional[List[int]] = None,
        max_batches: Optional[int] = None,
    ) -> MetricResult:
        """
        Evaluate model on training data.

        Args:
            model: Model to evaluate
            dataloader: Training data loader
            device: Device for evaluation
            epoch: Current epoch
            k_vals: K values for recall@K (currently unused)
            max_batches: Maximum batches to process

        Returns:
            MetricResult with training evaluation metrics
        """
        return self.train_evaluator.evaluate(
            model,
            feature_manager,
            embedding_manager,
            dataloader,
            device,
            epoch,
            k_vals,
            max_batches,
        )

    def evaluate_test(
        self,
        model,
        processor,
        dataloader: DataLoader,
        label_embeddings: torch.Tensor,
        epoch: int = 0,
        device: Optional[str] = None,
        return_detailed_results: bool = False,
    ) -> Union[MetricResult, Tuple]:
        """
        Evaluate model on test data.

        Args:
            model: Model to evaluate
            processor: Text processor for tokenization
            dataloader: Test data loader
            label_embeddings: Available label embeddings for oracle
            epoch: Current epoch
            device: Device for evaluation
            inspect_labels: Whether to return detailed analysis
            use_best_label: Whether to use best label for oracle

        Returns:
            MetricResult
        """
        return self.test_evaluator.evaluate(
            model,
            processor,
            dataloader,
            label_embeddings,
            epoch,
            device,
            return_detailed_results,
        )

    def encode_test_data(
        self, model, processor, dataloader: DataLoader, device: Optional[str] = None
    ) -> Tuple:
        """
        Extract embeddings from test data without evaluation.

        Args:
            model: Model to use for encoding
            processor: Text processor
            dataloader: Test data loader
            device: Device for encoding

        Returns:
            Tuple with embeddings and mappings
        """
        return self.test_evaluator.encode_data_only(
            model, processor, dataloader, device
        )

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

        # Update evaluator configs
        self.train_evaluator.config = self.config
        self.test_evaluator.config = self.config


# Backward compatibility functions
def inference_train(
    model,
    dataloader: DataLoader,
    feature_manager,
    embedding_manager,
    device: str,
    epoch: int = 0,
    k_vals: List[int] = [1, 5, 10],
    max_batches: int = 25,
) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for training evaluation.

    This function maintains the original interface from inference.py
    while using the new refactored implementation.
    """
    config = EvaluationConfig(
        device=device,
        train_max_batches=max_batches,
        k_vals=k_vals or [1, 5, 10],
        print_metrics=True,
    )

    manager = EvaluationManager(config)
    result = manager.evaluate_train(
        model,
        feature_manager,
        embedding_manager,
        dataloader,
        device,
        epoch,
        k_vals,
        max_batches,
    )

    return result.metrics


def inference_test(
    model,
    processor,
    dataloader: DataLoader,
    label_embeddings: torch.Tensor,
    epoch: int,
    device: str,
) -> Union[Dict[str, Any], Tuple]:
    """
    Backward compatibility wrapper for test evaluation.

    This function maintains the original interface from inference.py
    while using the new refactored implementation.
    """
    config = EvaluationConfig(
        device=device,
        print_metrics=True,
    )

    manager = EvaluationManager(config)
    result = manager.evaluate_test(
        model,
        processor,
        dataloader,
        label_embeddings,
        epoch,
        device,
    )

    return result.metrics


def encode_data(model, processor, dataloader: DataLoader, device: str) -> Tuple:
    """
    Backward compatibility wrapper for data encoding.

    This function maintains the original interface from inference.py.
    """
    config = EvaluationConfig(device=device)
    manager = EvaluationManager(config)

    return manager.encode_test_data(model, processor, dataloader, device)
