"""Test set evaluator."""

from typing import Optional, Dict, Any, Tuple, Union, List
import torch
from torch.utils.data import DataLoader

from .base import BaseEvaluator
from ..config import EvaluationConfig, MetricResult
from ..metrics import RecallMetrics, OracleMetrics


class TestEvaluator(BaseEvaluator):
    """Evaluator for test dataset using full recall evaluation."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
        self.recall_metrics = RecallMetrics(self.config)
        self.oracle_metrics = OracleMetrics(self.config)

    def evaluate(
        self,
        model,
        processor,
        dataloader: DataLoader,
        label_embeddings: Optional[torch.Tensor] = None,
        epoch: int = 0,
        device: Optional[str] = None,
        return_detailed_results: bool = False,
        use_oracle: bool = False,
    ) -> Union[MetricResult, Tuple]:
        """
        Evaluate model on test dataset with oracle and raw metrics.

        Args:
            model: The model to evaluate
            processor: Text processor for tokenization
            dataloader: Test data loader
            label_embeddings: Available label embeddings for oracle evaluation
            epoch: Current epoch number
            device: Device to run evaluation on
            inspect_labels: Whether to return detailed label analysis
            use_best_label: Whether to use best or first label for oracle

        Returns:
            MetricResult or tuple with detailed analysis if inspect_labels=True
        """
        device = device or self.config.device

        # Extract embeddings from test data
        print("Extracting embeddings from test data...")
        (
            all_img_emb,
            all_txt_emb,
            all_txt_full,
            all_raw_text,
            text_to_image_map,
            image_to_text_map,
        ) = self.processor.extract_embeddings(model, processor, dataloader)

        # Evaluate with oracle (trying all label embeddings)
        print("Running oracle evaluation...")

        if use_oracle:
            metrics_oracle, best_label_tti, best_label_itt = (
                self.oracle_metrics.compute_oracle_recall(
                    model,
                    label_embeddings,
                    all_img_emb,
                    all_txt_emb,
                    all_txt_full,
                    text_to_image_map,
                    image_to_text_map,
                    "oracle",
                )
            )
        else:
            metrics_oracle_txt, best_label_tti, best_label_itt = (
                self.oracle_metrics.compute_non_oracle_recall(
                    model,
                    label_embeddings,
                    all_img_emb,
                    all_txt_emb,
                    all_txt_full,
                    text_to_image_map,
                    image_to_text_map,
                    "txt_non_oracle",
                )
            )

            metrics_oracle, _, _ = self.oracle_metrics.compute_non_oracle_recall_imgtxt(
                model,
                label_embeddings,
                all_img_emb,
                all_txt_emb,
                all_txt_full,
                text_to_image_map,
                image_to_text_map,
                "imgtxt_non_oracle",
            )

        # Evaluate raw embeddings (without labels)
        print("Running raw evaluation...")
        metrics_raw = self.recall_metrics.compute_all_recalls(
            all_img_emb, all_txt_emb, text_to_image_map, image_to_text_map, "raw"
        )

        # Compute difference metrics
        metrics_diff = self._compute_metric_difference(
            metrics_oracle, metrics_raw, "raw", "diff"
        )

        # Combine all metrics
        if use_oracle:
            all_metrics = {
                "test/epoch": epoch,
                **metrics_oracle,
                **metrics_raw,
                **metrics_diff,
            }

        else:
            all_metrics = {
                "test/epoch": epoch,
                **metrics_oracle,
                **metrics_oracle_txt,
                **metrics_raw,
                **metrics_diff,
            }

        results = self._format_results(all_metrics, epoch)

        if self.config.print_metrics:
            self._print_results(results)

        return (
            results
            if not return_detailed_results
            else self._create_detailed_results(
                all_img_emb,
                all_txt_emb,
                all_txt_full,
                all_raw_text,
                text_to_image_map,
                image_to_text_map,
                best_label_tti,
                best_label_itt,
                results,
            )
        )

    def _compute_metric_difference(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float],
        prefix2: str,
        new_prefix: str,
    ) -> Dict[str, float]:
        """Compute difference between two metric dictionaries."""
        metric_diff = {}

        for key in metrics1:
            if "/" not in key:
                continue

            # Extract metric name after prefix
            metric_name = key.split("/", 1)[1]
            corresponding_key = f"{prefix2}/{metric_name}"

            if corresponding_key in metrics2:
                diff_key = f"{new_prefix}/{metric_name}"
                metric_diff[diff_key] = metrics1[key] - metrics2[corresponding_key]

        return metric_diff

    def _create_detailed_results(
        self,
        all_img_emb: torch.Tensor,
        all_txt_emb: torch.Tensor,
        all_txt_full: torch.Tensor,
        all_raw_text: List[str],
        text_to_image_map: torch.Tensor,
        image_to_text_map: torch.Tensor,
        best_label_tti: torch.Tensor,
        best_label_itt: torch.Tensor,
        results: MetricResult,
    ) -> Tuple:
        """Create detailed results for label inspection."""

        # # Compute raw ranking indices for additional analysis
        # img_emb_norm = self.processor.normalize_embeddings(all_img_emb)
        # txt_emb_norm = self.processor.normalize_embeddings(all_txt_emb)

        # dist_matrix_raw = img_emb_norm @ txt_emb_norm.T
        # inds_raw_itt = torch.argsort(dist_matrix_raw, dim=1, descending=True)
        # inds_raw_tti = torch.argsort(dist_matrix_raw.T, dim=1, descending=True)

        # print(
        #     "The order of returned tuple is all_img_emb, all_txt_emb, all_txt_full, text_to_image_map, image_to_text_map, best_label_tti, best_label_itt, inds_raw_tti, inds_raw_itt, results"
        # )

        return (
            all_img_emb,
            all_txt_emb,
            all_raw_text,
            # all_txt_full,
            text_to_image_map,
            image_to_text_map,
            # best_label_tti,
            # best_label_itt,
            # inds_raw_tti,
            # inds_raw_itt,
            results,
        )

    def encode_data_only(
        self, model, processor, dataloader: DataLoader, device: Optional[str] = None
    ) -> Tuple:
        """Extract embeddings and mappings without evaluation.

        Useful for getting encoded data for external analysis.
        """
        device = device or self.config.device

        # Extract embeddings
        (
            all_img_emb,
            all_txt_emb,
            all_txt_full,
            all_raw_text,
            text_to_image_map,
            image_to_text_map,
        ) = self.processor.extract_embeddings(model, processor, dataloader)

        # Compute raw ranking indices
        img_emb_norm = self.processor.normalize_embeddings(all_img_emb)
        txt_emb_norm = self.processor.normalize_embeddings(all_txt_emb)

        dist_matrix_raw = img_emb_norm @ txt_emb_norm.T
        inds_raw_itt = torch.argsort(dist_matrix_raw, dim=1, descending=True)
        inds_raw_tti = torch.argsort(dist_matrix_raw.T, dim=1, descending=True)

        return (
            all_img_emb,
            all_txt_emb,
            all_txt_full,
            all_raw_text,
            text_to_image_map,
            image_to_text_map,
            inds_raw_tti,
            inds_raw_itt,
        )
