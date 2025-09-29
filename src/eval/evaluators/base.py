"""Base evaluator class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

from ..config import EvaluationConfig, MetricResult
from ..processors import EmbeddingProcessor


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.processor = EmbeddingProcessor(self.config)
        
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> MetricResult:
        """Perform evaluation and return results."""
        pass
        
    def _format_results(self, metrics: Dict[str, float], epoch: Optional[int] = None,
                       additional_data: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Format evaluation results."""
        if epoch is not None:
            metrics = {**metrics, f"{self.config.metric_prefix}epoch": epoch}
            
        return MetricResult(
            metrics=metrics,
            epoch=epoch,
            additional_data=additional_data
        )
        
    def _print_results(self, results: MetricResult, title: str = "Evaluation Results"):
        """Print evaluation results in a formatted way."""
        if not self.config.print_metrics:
            return
            
        print(f"\n{'=' * 50}")
        print(f"{title}")
        if results.epoch is not None:
            print(f"Epoch: {results.epoch}")
        print(f"{'=' * 50}")
        
        for key, value in results.metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
        print(f"{'=' * 50}\n")