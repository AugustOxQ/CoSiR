"""Test set evaluator."""

from typing import Optional, Dict, Any, Tuple, Union, List
import torch
from torch.utils.data import DataLoader

from .base import BaseEvaluator
from ..config import EvaluationConfig, MetricResult, TestEvaluationDetail
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
        oracle_aggregation: str = "max",
    ) -> Union[MetricResult, TestEvaluationDetail]:
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
                self.oracle_metrics.compute_oracle_recall_average(
                    model,
                    label_embeddings,
                    all_img_emb,
                    all_txt_emb,
                    all_txt_full,
                    text_to_image_map,
                    image_to_text_map,
                    "oracle",
                    aggregation=oracle_aggregation,
                )
            )
        else:
            metrics_oracle, _, _ = self.oracle_metrics.compute_non_oracle_recall_txt(
                model,
                label_embeddings,
                all_img_emb,
                all_txt_emb,
                all_txt_full,
                text_to_image_map,
                image_to_text_map,
                "txt_non_oracle",
            )

            metrics_oracle_img, _, _ = (
                self.oracle_metrics.compute_non_oracle_recall_img(
                    model,
                    label_embeddings,
                    all_img_emb,
                    all_txt_emb,
                    all_txt_full,
                    text_to_image_map,
                    image_to_text_map,
                    "img_non_oracle",
                )
            )

            metrics_oracle_imgtxt, best_label_tti, best_label_itt = (
                self.oracle_metrics.compute_non_oracle_recall_imgtxt(
                    model,
                    label_embeddings,
                    all_img_emb,
                    all_txt_emb,
                    all_txt_full,
                    text_to_image_map,
                    image_to_text_map,
                    "both_non_oracle",
                )
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

        # Each group gets its own top-level wandb section (test_oracle/*, test_raw/*, …)
        # so the UI shows one panel per group rather than one crowded "test" panel.
        # The input dicts already carry their own group prefix (e.g. "oracle/i2t/R@1"),
        # so strip it before prepending the wandb section name.
        def _group(d: dict, group: str) -> dict:
            pfx = f"{group}/"
            return {
                f"test_{group}/{k[len(pfx):]}" if k.startswith(pfx) else f"test_{group}/{k}": v
                for k, v in d.items()
            }

        if use_oracle:
            all_metrics = {
                **_group(metrics_oracle, "oracle"),
                **_group(metrics_raw, "raw"),
                **_group(metrics_diff, "diff"),
            }
        else:
            all_metrics = {
                **_group(metrics_oracle, "oracle"),
                **_group(metrics_oracle_img, "oracle_img"),
                **_group(metrics_oracle_imgtxt, "oracle_imgtxt"),
                **_group(metrics_raw, "raw"),
                **_group(metrics_diff, "diff"),
            }

        results = self._format_results(all_metrics, epoch)

        if self.config.print_metrics:
            self._print_results(results)

        if not return_detailed_results:
            return results
        return TestEvaluationDetail(
            results=results,
            all_img_emb=all_img_emb,
            all_txt_emb=all_txt_emb,
            all_raw_text=all_raw_text,
            text_to_image_map=text_to_image_map,
            image_to_text_map=image_to_text_map,
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
