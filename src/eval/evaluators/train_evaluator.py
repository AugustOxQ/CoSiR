"""Training set evaluator."""

from typing import List, Optional, Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseEvaluator
from ..config import EvaluationConfig, MetricResult
from ..utils.label_utils import replace_with_most_different


class TrainEvaluator(BaseEvaluator):
    """Evaluator for training dataset using batch-wise similarity comparison."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)

    def evaluate(
        self,
        model,
        feature_manager,
        embedding_manager,
        dataloader: DataLoader,
        device,
        epoch: int = 0,
        k_vals: Optional[List[int]] = None,
        max_batches: Optional[int] = None,
    ) -> MetricResult:
        """
        Evaluate model on training data using batch-wise similarity ranking.

        Args:
            model: The model to evaluate
            dataloader: Training data loader
            device: Device to run evaluation on
            epoch: Current epoch number
            k_vals: List of K values for recall@K (unused in current implementation)
            max_batches: Maximum number of batches to process

        Returns:
            MetricResult containing evaluation metrics
        """
        device = device or self.config.device
        max_batches = max_batches or self.config.train_max_batches

        model.eval()

        total_rank_raw = 0.0
        total_rank_comb = 0.0
        total_rank_comb_shuffled = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_id, batch in enumerate(
                tqdm(dataloader, desc="Training evaluation")
            ):
                if batch_id >= max_batches:
                    print(
                        f"Epoch {epoch}: Stopping inference after {max_batches} batches"
                    )
                    break

                # Use batch_idx as chunk_id for direct chunk loading
                chunk_id = batch_id

                # Load features by chunk directly
                features_data = feature_manager.get_features_by_chunk(chunk_id)

                img_features = features_data["img_features"].to(
                    device, non_blocking=True
                )
                txt_features = features_data["txt_features"].to(
                    device, non_blocking=True
                )
                txt_full = features_data["txt_full"].to(device, non_blocking=True)
                batch_sample_ids = features_data["sample_ids"]

                # Get embeddings by chunk directly from optimized manager
                chunk_sample_ids, label_embeddings_data = (
                    embedding_manager.get_embeddings_by_chunk(chunk_id)
                )

                label_embeddings = label_embeddings_data.to(device)

                comb_emb, label_embedding_proj = model.combine(
                    txt_features,
                    txt_full,
                    label_embeddings,
                    epoch=epoch,
                    return_label_proj=True,
                )

                label_embedding_neg = replace_with_most_different(label_embeddings)

                comb_emb_neg = model.combine(
                    txt_features,
                    txt_full,
                    label_embedding_neg,
                    epoch=epoch,
                    return_label_proj=False,
                )

                # Move to CPU for similarity computation
                img_emb_cpu = img_features.cpu().numpy()
                txt_emb_cls_cpu = txt_features.cpu().numpy()
                comb_emb_cpu = comb_emb.cpu().numpy()
                comb_emb_shuffled_cpu = comb_emb_neg.cpu().numpy()

                # Compute similarity matrices
                sim_raw = cosine_similarity(img_emb_cpu, txt_emb_cls_cpu)
                sim_comb = cosine_similarity(img_emb_cpu, comb_emb_cpu)
                sim_comb_shuffled = cosine_similarity(
                    img_emb_cpu, comb_emb_shuffled_cpu
                )

                # Compute mean ranks for this batch
                avg_rank_raw = self._compute_ranks(sim_raw)
                avg_rank_comb = self._compute_ranks(sim_comb)
                avg_rank_comb_shuffled = self._compute_ranks(sim_comb_shuffled)

                total_rank_raw += avg_rank_raw
                total_rank_comb += avg_rank_comb
                total_rank_comb_shuffled += avg_rank_comb_shuffled
                total_samples += 1

                # Cleanup
                del (
                    img_features,
                    txt_features,
                    txt_full,
                    label_embeddings,
                    comb_emb,
                    comb_emb_neg,
                    label_embedding_neg,
                )
                torch.cuda.empty_cache()

        # Compute final metrics
        mean_rank_raw = total_rank_raw / total_samples
        mean_rank_comb = total_rank_comb / total_samples
        mean_rank_comb_shuffled = total_rank_comb_shuffled / total_samples

        metrics = {
            "val/mean_rank_raw": mean_rank_raw,
            "val/mean_rank_comb": mean_rank_comb,
            "val/mean_rank_comb_shuffled": mean_rank_comb_shuffled,
        }

        results = self._format_results(metrics, epoch)

        if self.config.print_metrics:
            print(
                f"Epoch {epoch}: Mean Rank - Raw: {mean_rank_raw:.2f}, "
                f"Combined: {mean_rank_comb:.2f}, Shuffled: {mean_rank_comb_shuffled:.2f}"
            )

        return results

    def _compute_ranks(self, similarity_matrix: np.ndarray) -> float:
        """Compute the mean rank of diagonal elements in similarity matrix."""
        ranks = []
        for i in range(similarity_matrix.shape[0]):
            row = similarity_matrix[i]
            sorted_indices = np.argsort(row)[::-1]  # Descending order
            rank = np.where(sorted_indices == i)[0][0] + 1  # 1-based rank
            ranks.append(rank)
        return float(np.mean(ranks))
