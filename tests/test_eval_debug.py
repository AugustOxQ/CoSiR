#!/usr/bin/env python3
"""Comprehensive test script for the refactored evaluation system."""

import sys
import traceback
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/project/CoSiR')

def test_imports():
    """Test basic imports to identify missing dependencies."""
    print("🧪 Testing basic imports...")

    try:
        print("  - Testing config import...")
        from src.eval.config import EvaluationConfig, MetricResult
        print("    ✅ Config import successful")
    except Exception as e:
        print(f"    ❌ Config import failed: {e}")
        return False

    try:
        print("  - Testing utils import...")
        from src.eval.utils.label_utils import replace_with_most_different
        print("    ✅ Utils import successful")
    except Exception as e:
        print(f"    ❌ Utils import failed: {e}")
        return False

    try:
        print("  - Testing metrics import...")
        from src.eval.metrics.base import BaseMetric, RankingMetric
        from src.eval.metrics.recall import RecallMetrics, OracleMetrics
        print("    ✅ Metrics import successful")
    except Exception as e:
        print(f"    ❌ Metrics import failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("  - Testing processors import...")
        from src.eval.processors.embeddings import EmbeddingProcessor
        print("    ✅ Processors import successful")
    except Exception as e:
        print(f"    ❌ Processors import failed: {e}")
        return False

    try:
        print("  - Testing evaluators import...")
        from src.eval.evaluators.base import BaseEvaluator
        from src.eval.evaluators.train_evaluator import TrainEvaluator
        from src.eval.evaluators.test_evaluator import TestEvaluator
        print("    ✅ Evaluators import successful")
    except Exception as e:
        print(f"    ❌ Evaluators import failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("  - Testing interface import...")
        from src.eval.interface import EvaluationManager, inference_train, inference_test
        print("    ✅ Interface import successful")
    except Exception as e:
        print(f"    ❌ Interface import failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("  - Testing main package import...")
        from src.eval import EvaluationManager, EvaluationConfig
        print("    ✅ Main package import successful")
    except Exception as e:
        print(f"    ❌ Main package import failed: {e}")
        traceback.print_exc()
        return False

    return True

def test_config():
    """Test configuration classes."""
    print("🧪 Testing configuration...")

    try:
        from src.eval.config import EvaluationConfig, MetricResult

        # Test default config
        config = EvaluationConfig()
        print(f"    ✅ Default config created: device={config.device}, k_vals={config.k_vals}")

        # Test custom config
        config = EvaluationConfig(
            device="cpu",
            k_vals=[1, 5, 10, 20],
            batch_size=128
        )
        print(f"    ✅ Custom config created: device={config.device}, batch_size={config.batch_size}")

        # Test MetricResult
        metrics = {"test/accuracy": 0.85, "test/loss": 1.23}
        result = MetricResult(metrics=metrics, epoch=5)
        print(f"    ✅ MetricResult created: epoch={result.epoch}, metrics count={len(result.metrics)}")
        print(f"    ✅ Dict access works: accuracy={result['test/accuracy']}")

        return True

    except Exception as e:
        print(f"    ❌ Config test failed: {e}")
        traceback.print_exc()
        return False

def test_label_utils():
    """Test label utilities."""
    print("🧪 Testing label utilities...")

    try:
        from src.eval.utils.label_utils import replace_with_most_different

        # Create test embeddings
        embeddings = torch.randn(10, 64)  # 10 embeddings of 64 dims

        # Test replace_with_most_different
        new_embeddings = replace_with_most_different(embeddings, k=3)
        print(f"    ✅ replace_with_most_different works: input shape {embeddings.shape}, output shape {new_embeddings.shape}")

        # Verify they're different
        different = not torch.equal(embeddings, new_embeddings)
        print(f"    ✅ Embeddings are different: {different}")

        return True

    except Exception as e:
        print(f"    ❌ Label utils test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics():
    """Test metrics computation."""
    print("🧪 Testing metrics...")

    try:
        from src.eval.config import EvaluationConfig
        from src.eval.metrics.recall import RecallMetrics

        config = EvaluationConfig(device="cpu", k_vals=[1, 5])
        metrics = RecallMetrics(config)

        # Create dummy data
        img_emb = torch.randn(20, 128)  # 20 images, 128 dims
        txt_emb = torch.randn(100, 128)  # 100 texts (5 per image), 128 dims

        # Create mappings (5 texts per image)
        text_to_image_map = torch.tensor([i // 5 for i in range(100)])  # [0,0,0,0,0,1,1,1,1,1,...]
        image_to_text_map = torch.tensor([[i*5, i*5+1, i*5+2, i*5+3, i*5+4] for i in range(20)])  # [[0,1,2,3,4], [5,6,7,8,9], ...]

        # Test recall computation
        result = metrics.compute_all_recalls(
            img_emb, txt_emb, text_to_image_map, image_to_text_map, prefix="test"
        )

        print(f"    ✅ Recall metrics computed: {len(result)} metrics")
        print(f"    ✅ Sample metrics: i2t_R1={result.get('testi2t_R1', 'N/A')}, t2i_R1={result.get('testt2i_R1', 'N/A')}")

        return True

    except Exception as e:
        print(f"    ❌ Metrics test failed: {e}")
        traceback.print_exc()
        return False

def create_mock_model():
    """Create a mock model for testing."""

    class MockModule:
        def combine(self, txt_emb_cls, txt_emb, label_emb):
            # Simple combination: concatenate and project
            combined = torch.cat([txt_emb_cls, label_emb], dim=-1)
            # Project back to original size
            return torch.nn.functional.linear(combined, torch.randn(txt_emb_cls.size(-1), combined.size(-1)))

        def encode_img_txt(self, image_input, text_input):
            batch_size = image_input["pixel_values"].size(0)
            img_emb = torch.randn(batch_size, 512)
            txt_emb = torch.randn(batch_size, 512)
            txt_full = torch.randn(batch_size, 768)
            return img_emb, txt_emb, None, txt_full

    class MockModel:
        def __init__(self):
            self.module = MockModule()

        def eval(self):
            pass

    return MockModel()

def create_mock_dataloader(is_train=True):
    """Create a mock dataloader for testing."""

    class MockDataset:
        def __init__(self, is_train=True):
            self.is_train = is_train
            self.captions_per_image = 5

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            if self.is_train:
                # Training format: (img_emb, txt_emb_cls, txt_emb, label_embedding, sample_id)
                # The dataloader will stack these, so after squeezing we get (batch_size, feature_dim)
                return (
                    torch.randn(512),   # img_emb - will become (1, 512) after unsqueezing in dataloader
                    torch.randn(512),   # txt_emb_cls
                    torch.randn(768),   # txt_emb
                    torch.randn(64),    # label_embedding
                    idx                 # sample_id
                )
            else:
                # Test format: (image, raw_text)
                image = {"pixel_values": torch.randn(1, 3, 224, 224)}
                raw_text = [f"Caption {idx}_{i}" for i in range(5)]  # 5 captions per image
                return image, raw_text

    from torch.utils.data import DataLoader
    dataset = MockDataset(is_train)
    return DataLoader(dataset, batch_size=2, shuffle=False)

def test_train_evaluator():
    """Test TrainEvaluator functionality."""
    print("🧪 Testing TrainEvaluator...")

    try:
        from src.eval.evaluators.train_evaluator import TrainEvaluator
        from src.eval.config import EvaluationConfig

        config = EvaluationConfig(device="cpu", train_max_batches=2)
        evaluator = TrainEvaluator(config)

        model = create_mock_model()
        train_loader = create_mock_dataloader(is_train=True)

        # Test evaluation
        result = evaluator.evaluate(model, train_loader, device="cpu", epoch=1)

        print(f"    ✅ TrainEvaluator works: {len(result.metrics)} metrics returned")
        print(f"    ✅ Sample metrics: {list(result.metrics.keys())}")

        return True

    except Exception as e:
        print(f"    ❌ TrainEvaluator test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility functions."""
    print("🧪 Testing backward compatibility...")

    try:
        from src.eval.interface import inference_train

        model = create_mock_model()
        train_loader = create_mock_dataloader(is_train=True)

        # Test inference_train
        result = inference_train(model, train_loader, "cpu", epoch=1, max_batches=2)

        print(f"    ✅ inference_train works: {len(result)} metrics returned")
        print(f"    ✅ Sample metrics: {list(result.keys())}")

        return True

    except Exception as e:
        print(f"    ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 Comprehensive Evaluation System Debug\n")

    tests = [
        ("Basic Imports", test_imports),
        ("Configuration", test_config),
        ("Label Utils", test_label_utils),
        ("Metrics", test_metrics),
        ("TrainEvaluator", test_train_evaluator),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
            failed += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print('='*60)

    if failed == 0:
        print("🎉 All tests passed! The evaluation system is ready to use.")
    else:
        print(f"⚠️  {failed} tests failed. Issues need to be fixed.")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)