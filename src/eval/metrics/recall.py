"""Recall and ranking metrics implementation."""

from typing import Dict, List, Tuple
import torch
import numpy as np

from .base import RankingMetric
from ..config import EvaluationConfig


class RecallMetrics(RankingMetric):
    """Implements recall@K, mAP, and ranking metrics."""

    def compute(self, *args, **kwargs) -> Dict[str, float]:
        """Main compute method for compatibility with BaseMetric."""
        # This is a wrapper - actual computation is in specific methods
        return self.compute_all_recalls(*args, **kwargs)

    def compute_i2t_recall(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "i2t",
    ) -> Dict[str, float]:
        """Compute image-to-text recall metrics."""
        num_images = image_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        # Compute similarity matrix
        dist_matrix = text_embeddings @ image_embeddings.T
        if self.config.cpu_offload:
            dist_matrix = dist_matrix.cpu()
        dist_matrix = dist_matrix.T  # shape: [num_images, num_texts]

        # Sort in descending order
        inds = torch.argsort(dist_matrix, dim=1, descending=True)
        inds = inds.to(self.config.device)

        # Calculate recall@K
        recall_scores = []
        for k in self.config.k_vals:
            topk = inds[:, :k]
            correct = torch.zeros(
                (num_images,), dtype=torch.bool, device=self.config.device
            )

            # Check if any of the relevant captions was retrieved
            for i in range(captions_per_image):
                contains_index = torch.eq(
                    topk, image_to_text_map[:, i].unsqueeze(-1)
                ).any(dim=1)
                correct = torch.logical_or(correct, contains_index)

            num_correct = correct.sum().item()
            recall_scores.append(num_correct / num_images * 100)

        # Calculate additional metrics
        mean_rank, median_rank, mean_ap = self.calculate_recall_metrics(
            inds, image_to_text_map, captions_per_image
        )

        metrics = {}
        for i, k in enumerate(self.config.k_vals):
            metrics[f"{prefix}_R{k}"] = recall_scores[i]
        metrics.update(
            {
                f"{prefix}_meanR": mean_rank,
                f"{prefix}_medR": median_rank,
                f"{prefix}_mAP": mean_ap,
            }
        )

        return metrics

    def compute_t2i_recall(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_to_image_map: torch.Tensor,
        prefix: str = "t2i",
    ) -> Dict[str, float]:
        """Compute text-to-image recall metrics."""
        num_texts = text_embeddings.shape[0]

        # Compute similarity matrix
        dist_matrix = text_embeddings @ image_embeddings.T
        if self.config.cpu_offload:
            dist_matrix = dist_matrix.cpu()

        # Sort in descending order
        inds = torch.argsort(dist_matrix, dim=1, descending=True)
        inds = inds.to(self.config.device)

        # Calculate recall@K
        recall_scores = []
        for k in self.config.k_vals:
            topk = inds[:, :k]
            correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
            num_correct = correct.sum().item()
            recall_scores.append(num_correct / num_texts * 100)

        # Calculate additional metrics
        mean_rank, median_rank, mean_ap = self.calculate_recall_metrics(
            inds, text_to_image_map, 1
        )

        metrics = {}
        for i, k in enumerate(self.config.k_vals):
            metrics[f"{prefix}_R{k}"] = recall_scores[i]
        metrics.update(
            {
                f"{prefix}_meanR": mean_rank,
                f"{prefix}_medR": median_rank,
                f"{prefix}_mAP": mean_ap,
            }
        )

        return metrics

    def compute_all_recalls(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute both I2T and T2I recall metrics."""
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Compute both directions
        i2t_metrics = self.compute_i2t_recall(
            image_embeddings,
            text_embeddings,
            image_to_text_map,
            f"{prefix}/i2t" if prefix else "i2t",
        )
        t2i_metrics = self.compute_t2i_recall(
            image_embeddings,
            text_embeddings,
            text_to_image_map,
            f"{prefix}/t2i" if prefix else "t2i",
        )

        # Combine metrics
        all_metrics = {**i2t_metrics, **t2i_metrics}

        return all_metrics

    def _print_metrics(self, metrics: Dict[str, float], prefix: str):
        """Print metrics in a formatted way."""
        print(f"############start-{prefix}#########################")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print(f"############end-{prefix}#########################\n")


class OracleMetrics(RankingMetric):
    """Oracle evaluation that finds best label embedding for each query."""

    def compute(self, *args, **kwargs) -> Dict[str, float]:
        """Main compute method for compatibility with BaseMetric."""
        # Return the metrics part from oracle evaluation
        result = self.compute_oracle_recall(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    def compute_oracle_recall(
        self,
        model,
        label_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_full: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        prefix: str = "oracle",
    ) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
        """Compute oracle metrics by trying all label embeddings."""

        num_texts = text_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        num_labels = label_embeddings.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        # Move to device
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        image_embeddings = image_embeddings.to(self.config.device)
        text_embeddings = text_embeddings.to(self.config.device)
        text_full = text_full.to(self.config.device)

        # Initialize tracking variables
        best_inds_tti = None
        best_rank_tti = torch.full((num_texts,), float("inf"))
        best_label_tti = -torch.ones((num_texts,))

        best_inds_itt = None
        best_rank_itt = torch.full((num_images,), float("inf"))
        best_label_itt = -torch.ones((num_images,))

        print(f"Evaluating oracle with {num_labels} labels...")

        with torch.no_grad():
            for label_id in range(-1, num_labels):  # -1 for no label baseline
                if label_id == -1:
                    # Use raw text embeddings
                    combined_embeddings = text_embeddings.detach().clone()
                else:
                    # Combine with label embedding
                    label_emb = (
                        label_embeddings[label_id]
                        .expand(num_texts, -1)
                        .to(self.config.device)
                    )

                    # Process in batches to manage memory
                    combined_embs = []
                    for i in range(0, num_texts, self.config.batch_size):
                        end_idx = min(i + self.config.batch_size, num_texts)
                        batch_combined = model.combine(
                            text_embeddings[i:end_idx],
                            text_full[i:end_idx],
                            label_emb[i:end_idx],
                        ).detach()
                        combined_embs.append(batch_combined)
                    combined_embeddings = torch.cat(combined_embs, dim=0)

                combined_embeddings = combined_embeddings / combined_embeddings.norm(
                    dim=-1, keepdim=True
                )

                # Text-to-Image evaluation
                dist_matrix_tti = combined_embeddings @ image_embeddings.T
                if self.config.cpu_offload:
                    dist_matrix_tti = dist_matrix_tti.cpu()
                inds_tti = torch.argsort(dist_matrix_tti, dim=1, descending=True)

                # Calculate absolute ranking
                abs_rank_tti = self.absolute_rank(inds_tti, text_to_image_map, 1)
                abs_rank_tti = torch.tensor(abs_rank_tti, dtype=torch.float32)

                # Update best results
                if best_inds_tti is None:
                    best_inds_tti = inds_tti.clone()
                    best_rank_tti = abs_rank_tti.clone()
                else:
                    update_mask = abs_rank_tti < best_rank_tti
                    if update_mask.any():
                        best_inds_tti[update_mask] = inds_tti[update_mask]
                        best_rank_tti[update_mask] = abs_rank_tti[update_mask]
                        best_label_tti[update_mask] = label_id

                # Image-to-Text evaluation
                dist_matrix_itt = dist_matrix_tti.T
                if self.config.cpu_offload:
                    dist_matrix_itt = dist_matrix_itt.cpu()
                inds_itt = torch.argsort(dist_matrix_itt, dim=1, descending=True)

                # Calculate absolute ranking for I2T
                abs_rank_itt = self.absolute_rank(
                    inds_itt, image_to_text_map, captions_per_image
                )
                abs_rank_itt = torch.tensor(abs_rank_itt, dtype=torch.float32)

                # Use minimum rank per image for comparison
                abs_rank_itt_min = (
                    abs_rank_itt.view(-1, captions_per_image).min(dim=1).values
                )
                abs_rank_itt_sum = abs_rank_itt.view(-1, captions_per_image).sum(dim=1)
                abs_rank_itt_combined = abs_rank_itt_sum + abs_rank_itt_min

                # Update best results for I2T
                if best_inds_itt is None:
                    best_inds_itt = inds_itt.clone()
                    best_rank_itt = abs_rank_itt_combined.clone()
                else:
                    update_mask = abs_rank_itt_combined < best_rank_itt
                    if update_mask.any():
                        best_inds_itt[update_mask] = inds_itt[update_mask]
                        best_rank_itt[update_mask] = abs_rank_itt_combined[update_mask]
                        best_label_itt[update_mask] = label_id

                # Clean up memory
                del combined_embeddings
                torch.cuda.empty_cache()

        # Compute final metrics using best rankings
        metrics = self._compute_recall_from_indices(
            best_inds_tti,
            best_inds_itt,
            text_to_image_map,
            image_to_text_map,
            num_texts,
            num_images,
            captions_per_image,
            prefix,
        )

        return metrics, best_label_tti, best_label_itt

    def _compute_recall_from_indices(
        self,
        inds_tti: torch.Tensor,
        inds_itt: torch.Tensor,
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        num_texts: int,
        num_images: int,
        captions_per_image: int,
        prefix: str,
    ) -> Dict[str, float]:
        """Compute recall metrics from precomputed indices."""
        inds_tti = inds_tti.to(self.config.device)
        inds_itt = inds_itt.to(self.config.device)

        # Text-to-Image recall
        t2i_recall = []
        for k in self.config.k_vals:
            topk = inds_tti[:, :k]
            correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
            num_correct = correct.sum().item()
            t2i_recall.append(num_correct / num_texts)

        # Image-to-Text recall
        i2t_recall = []
        for k in self.config.k_vals:
            topk = inds_itt[:, :k]
            correct = torch.zeros(
                (num_images,), dtype=torch.bool, device=self.config.device
            )

            for i in range(captions_per_image):
                contains_index = torch.eq(
                    topk, image_to_text_map[:, i].unsqueeze(-1)
                ).any(dim=1)
                correct = torch.logical_or(correct, contains_index)

            num_correct = correct.sum().item()
            i2t_recall.append(num_correct / num_images)

        # Round i2t_recall and t2i_recall to 1 decimal places
        i2t_recall = [round(recall * 100, 1) for recall in i2t_recall]
        t2i_recall = [round(recall * 100, 1) for recall in t2i_recall]

        # Calculate additional metrics
        meanR_t2i, medR_t2i, mAP_t2i = self.calculate_recall_metrics(
            inds_tti, text_to_image_map, 1
        )
        meanR_i2t, medR_i2t, mAP_i2t = self.calculate_recall_metrics(
            inds_itt, image_to_text_map, captions_per_image
        )

        # Format results
        metrics = {}
        for i, k in enumerate(self.config.k_vals):
            metrics[f"{prefix}/i2t_R{k}"] = i2t_recall[i]
            metrics[f"{prefix}/t2i_R{k}"] = t2i_recall[i]

        metrics.update(
            {
                f"{prefix}/i2t_R1": i2t_recall[0],
                f"{prefix}/i2t_R5": i2t_recall[1],
                f"{prefix}/i2t_R10": i2t_recall[2],
                f"{prefix}/i2t_meanR": meanR_i2t,
                f"{prefix}/i2t_medR": medR_i2t,
                f"{prefix}/i2t_mAP": mAP_i2t,
                f"{prefix}/t2i_R1": t2i_recall[0],
                f"{prefix}/t2i_R5": t2i_recall[1],
                f"{prefix}/t2i_R10": t2i_recall[2],
                f"{prefix}/t2i_meanR": meanR_t2i,
                f"{prefix}/t2i_medR": medR_t2i,
                f"{prefix}/t2i_mAP": mAP_t2i,
            }
        )

        return metrics
