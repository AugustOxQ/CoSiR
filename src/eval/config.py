"""Configuration classes for evaluation system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class EvaluationConfig:
    """Configuration for evaluation system."""

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Metric configuration
    k_vals: List[int] = field(default_factory=lambda: [1, 5, 10])

    # Memory management
    batch_size: int = 512
    max_batches: Optional[int] = 25
    cpu_offload: bool = True  # Whether to move large matrices to CPU for sorting

    # Text processing
    max_text_length: int = 77

    # Training evaluation specific
    train_max_batches: int = 25

    # Oracle evaluation
    oracle_topk_different: int = 10  # k for most different embeddings

    evaluation_interval: int = 1

    # Logging
    print_metrics: bool = True
    metric_prefix: str = ""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.k_vals:
            raise ValueError("k_vals cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_text_length <= 0:
            raise ValueError("max_text_length must be positive")


@dataclass
class MetricResult:
    """Container for evaluation metric results."""

    metrics: Dict[str, float]
    epoch: Optional[int] = None
    additional_data: Optional[Dict[str, Any]] = None

    def __getitem__(self, key: str) -> float:
        """Allow dict-like access to metrics."""
        return self.metrics[key]

    def get(self, key: str, default: float = 0.0) -> float:
        """Get metric with default value."""
        return self.metrics.get(key, default)

    def update(self, other_metrics: Dict[str, float]) -> None:
        """Update metrics with new values."""
        self.metrics.update(other_metrics)
