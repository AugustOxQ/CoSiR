"""Refactored evaluation package with improved organization and APIs.

This package provides a comprehensive evaluation system for multimodal models
with the following key features:

- Configurable evaluation pipeline
- Modular metric computation
- Efficient memory management  
- Backward compatibility with original interfaces
- Clean separation of concerns

Key Components:
- EvaluationManager: Unified interface for all evaluation tasks
- TrainEvaluator: Training set evaluation with batch-wise ranking
- TestEvaluator: Test set evaluation with oracle and raw metrics
- RecallMetrics: Recall@K, mAP, and ranking metric computation
- OracleMetrics: Oracle evaluation trying all label embeddings

Usage Examples:

# New API (recommended)
from src.eval import EvaluationManager, EvaluationConfig

config = EvaluationConfig(device="cuda", k_vals=[1, 5, 10])
manager = EvaluationManager(config)

# Training evaluation
train_results = manager.evaluate_train(model, train_loader, epoch=epoch)

# Test evaluation  
test_results = manager.evaluate_test(
    model, processor, test_loader, label_embeddings, epoch=epoch
)

# Backward compatibility (original interface)
from src.eval import inference_train, inference_test

train_log = inference_train(model, train_loader, device, epoch, [1, 5, 10])
test_log = inference_test(model, processor, test_loader, representatives, epoch, device)
"""

# Main interfaces
from .interface import EvaluationManager, inference_train, inference_test, encode_data
from .config import EvaluationConfig, MetricResult

# Evaluators
from .evaluators import TrainEvaluator, TestEvaluator

# Metrics  
from .metrics import RecallMetrics, OracleMetrics

# Utilities
from .utils import replace_with_most_different, sample_label_embeddings

__all__ = [
    # Main interfaces
    'EvaluationManager',
    'EvaluationConfig', 
    'MetricResult',
    
    # Evaluators
    'TrainEvaluator',
    'TestEvaluator',
    
    # Metrics
    'RecallMetrics', 
    'OracleMetrics',
    
    # Utilities
    'replace_with_most_different',
    'sample_label_embeddings',
    
    # Backward compatibility
    'inference_train',
    'inference_test', 
    'encode_data',
]