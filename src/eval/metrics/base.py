"""Base classes for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch
import numpy as np

from ..config import EvaluationConfig, MetricResult


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, float]:
        """Compute the metric."""
        pass

    def _format_metrics(
        self, metrics: Dict[str, float], prefix: str = ""
    ) -> Dict[str, float]:
        """Format metrics with optional prefix."""
        if not prefix:
            prefix = self.config.metric_prefix

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        return {f"{prefix}{key}": value for key, value in metrics.items()}


class RankingMetric(BaseMetric):
    """Base class for ranking-based metrics."""

    def calculate_average_precision(
        self, correct_positions: torch.Tensor, total_relevant: int
    ) -> float:
        """Calculate Average Precision (AP) for the given ranks of relevant documents."""
        if total_relevant == 0 or correct_positions.numel() == 0:
            return 0.0

        ap_sum = 0.0
        for i, rank in enumerate(correct_positions.sort()[0], 1):
            precision_at_rank = i / float(rank + 1)  # Correct for 1-based indexing
            ap_sum += precision_at_rank

        return ap_sum / total_relevant

    def absolute_rank(
        self, inds: torch.Tensor, mappings: torch.Tensor, captions_per_image: int
    ) -> List[int]:
        """Calculate absolute ranks for evaluation."""
        num_queries = inds.size(0)
        all_ranks = []

        for query_idx in range(num_queries):
            correct_indices = mappings[query_idx].tolist()
            query_inds = inds[query_idx]

            if isinstance(correct_indices, int):
                # Single correct index
                correct_mask = query_inds == torch.tensor(
                    correct_indices, device=query_inds.device
                )
                correct_positions = correct_mask.nonzero(as_tuple=True)[-1].item()
                ranks = correct_positions + 1  # Convert to 1-based indexing
            else:
                # Multiple correct indices
                ranks = []
                for correct_index in correct_indices:
                    position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                    rank = position.item() + 1
                    ranks.append(rank)
                assert len(ranks) == captions_per_image

            if not isinstance(ranks, list):
                ranks = [ranks]
            all_ranks.extend(ranks)

        return all_ranks

    def calculate_recall_metrics(
        self, inds: torch.Tensor, mappings: torch.Tensor, captions_per_image: int
    ) -> Tuple[float, float, float]:
        """Calculate mean rank, median rank, and mAP."""
        num_queries = inds.size(0)
        AP_scores = []
        all_ranks = []

        for query_idx in range(num_queries):
            correct_indices = mappings[query_idx].tolist()
            query_inds = inds[query_idx]

            if isinstance(correct_indices, int):
                # Single correct index
                correct_mask = query_inds == torch.tensor(
                    correct_indices, device=self.config.device
                )
                correct_positions = correct_mask.nonzero(as_tuple=True)[-1].item()
                ranks = correct_positions + 1
            else:
                # Multiple correct indices
                ranks = []
                for correct_index in correct_indices:
                    position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                    rank = position.item() + 1
                    ranks.append(rank)
                assert len(ranks) == captions_per_image

            if not isinstance(ranks, list):
                ranks = [ranks]
            all_ranks.extend(ranks)

            # Calculate AP for this query
            AP = 0
            for j, rank in enumerate(sorted(ranks), start=1):
                precision_at_j = j / rank
                AP += precision_at_j
            AP /= captions_per_image
            AP_scores.append(AP)

        mean_rank = float(np.mean(all_ranks))
        median_rank = float(np.median(all_ranks))
        mean_ap = float(np.mean(AP_scores))

        # Round rank to integer
        mean_rank = int(mean_rank)
        median_rank = int(median_rank)

        # Round AP to 2 decimal places, with percentage
        mean_ap = round(mean_ap * 100, 1)

        return mean_rank, median_rank, mean_ap
