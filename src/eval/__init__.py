"""Evaluation package for CoSiR training and test-set retrieval metrics."""

from .config import EvaluationConfig, MetricResult, TestEvaluationDetail
from .pipeline import (
    EvaluationManager,
    TrainEvaluator,
    TestEvaluator,
    replace_with_most_different,
    sample_label_embeddings,
    inference_train,
    inference_test,
    encode_data,
)
from .metrics import RecallMetrics, OracleMetrics
