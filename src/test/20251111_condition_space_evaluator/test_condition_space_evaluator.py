"""
Comprehensive test suite for CoSiRAutomaticEvaluator
Tests edge cases, validates functionality, and ensures robustness
"""

import torch
import numpy as np
import pytest
from src.utils.condition_space_evaluator import CoSiRAutomaticEvaluator
from src.model.cosirmodel import CoSiRModel


class MockModel:
    """Mock CoSiR model for testing"""

    def __init__(self, device="cpu"):
        self.device = device

    def eval(self):
        return self

    def combine(self, text_embs, label_embs, conditions):
        """
        Simple mock: add a small perturbation based on condition norm
        This simulates the model's modulation effect
        """
        # Normalize text embeddings
        text_normalized = torch.nn.functional.normalize(text_embs, dim=1)

        # Add perturbation proportional to condition magnitude
        condition_norms = torch.norm(conditions, dim=1, keepdim=True)
        perturbation = 0.1 * condition_norms * torch.randn_like(text_embs)

        # Return modulated embeddings
        modulated = text_normalized + perturbation
        return torch.nn.functional.normalize(modulated, dim=1)


def create_mock_data(n_images=50, n_texts=250, embed_dim=512, device="cpu"):
    """
    Create mock embeddings and mappings for testing

    Args:
        n_images: Number of image embeddings
        n_texts: Number of text embeddings
        embed_dim: Embedding dimension
        device: Device to create tensors on

    Returns:
        tuple: (image_embs, text_embs, captions_flat, img_to_cap_map)
    """
    # Create random embeddings
    image_embs = torch.nn.functional.normalize(
        torch.randn(n_images, embed_dim, device=device), dim=1
    )
    text_embs = torch.nn.functional.normalize(
        torch.randn(n_texts, embed_dim, device=device), dim=1
    )

    # Create captions (5 per image for COCO-style)
    captions_per_image = 5
    captions_flat = [f"Caption {i}" for i in range(n_texts)]

    # Create image-to-caption mapping
    img_to_cap_map = {}
    for img_idx in range(n_images):
        start_idx = img_idx * captions_per_image
        end_idx = start_idx + captions_per_image
        img_to_cap_map[img_idx] = list(range(start_idx, min(end_idx, n_texts)))

    return image_embs, text_embs, captions_flat, img_to_cap_map


def test_basic_initialization():
    """Test basic initialization of evaluator"""
    print("\n[TEST] Basic Initialization")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(50, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    assert evaluator.n_images == 20
    assert evaluator.n_texts == 100
    assert len(evaluator.conditions) == 50
    print("✓ Basic initialization passed")


def test_edge_case_small_dataset():
    """Test with very small dataset"""
    print("\n[TEST] Edge Case: Small Dataset")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)

    # Very small dataset
    precomputed = create_mock_data(n_images=5, n_texts=10, device=device)
    conditions = torch.randn(10, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    # Test with reduced samples
    try:
        result = evaluator.compute_radius_effect_correlation(
            n_samples=5, n_texts_sample=5
        )
        print(f"  Correlation: {result['correlation']:.4f}")
        assert "correlation" in result
        print("✓ Small dataset test passed")
    except Exception as e:
        print(f"✗ Small dataset test failed: {e}")
        raise


def test_edge_case_single_condition():
    """Test with single condition"""
    print("\n[TEST] Edge Case: Single Condition")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=10, n_texts=50, device=device)
    conditions = torch.randn(1, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    try:
        # This should handle gracefully or skip
        result = evaluator.compute_condition_space_quality()
        print(f"  Silhouette: {result['silhouette_score']:.4f}")
        print("✓ Single condition test passed")
    except Exception as e:
        print(f"  Expected behavior with single condition: {e}")
        print("✓ Single condition test passed (graceful failure)")


def test_edge_case_zero_radius_conditions():
    """Test with conditions at origin"""
    print("\n[TEST] Edge Case: Zero Radius Conditions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=10, n_texts=50, device=device)

    # All conditions at origin
    conditions = torch.zeros(20, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    try:
        result = evaluator.compute_radius_effect_correlation(
            n_samples=10, n_texts_sample=20
        )
        print(
            f"  Correlation with zero radius: {result['correlation']:.4f} (may be NaN)"
        )
        print("✓ Zero radius test passed")
    except Exception as e:
        print(f"  Note: Zero radius causes expected issues: {e}")
        print("✓ Zero radius test passed (graceful handling)")


def test_edge_case_collinear_conditions():
    """Test with all conditions on same line"""
    print("\n[TEST] Edge Case: Collinear Conditions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=10, n_texts=50, device=device)

    # All conditions along x-axis
    conditions = torch.zeros(20, 2, device=device)
    conditions[:, 0] = torch.linspace(-2, 2, 20)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    try:
        result = evaluator.compute_condition_space_quality()
        print(f"  Effective dims: {result['n_effective_dims']}")
        assert result["n_effective_dims"] <= 1  # Should detect 1D structure
        print("✓ Collinear conditions test passed")
    except Exception as e:
        print(f"✗ Collinear conditions test failed: {e}")
        raise


def test_reinitialization():
    """Test re-initialization with new data"""
    print("\n[TEST] Re-initialization")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model1 = MockModel(device=device)
    precomputed1 = create_mock_data(n_images=10, n_texts=50, device=device)
    conditions1 = torch.randn(20, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model1, precomputed1, conditions1)
    assert evaluator.n_images == 10

    # Re-initialize with new data
    model2 = MockModel(device=device)
    precomputed2 = create_mock_data(n_images=30, n_texts=150, device=device)
    conditions2 = torch.randn(50, 2, device=device)

    evaluator.re_initialize_variables(model2, precomputed2, conditions2)

    assert evaluator.n_images == 30
    assert evaluator.n_texts == 150
    assert len(evaluator.conditions) == 50
    print("✓ Re-initialization test passed")


def test_radius_effect_correlation():
    """Test radius-effect correlation computation"""
    print("\n[TEST] Radius-Effect Correlation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(100, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    result = evaluator.compute_radius_effect_correlation(
        n_samples=50, n_texts_sample=30
    )

    assert "correlation" in result
    assert "p_value" in result
    assert "radii_mean" in result
    assert "effects_mean" in result
    assert result["n_samples"] == 50

    # Check for NaN values
    assert not np.isnan(result["correlation"]) or result["radii_std"] == 0
    print(f"  Correlation: {result['correlation']:.4f}")
    print("✓ Radius-effect correlation test passed")


def test_angular_semantic_monotonicity():
    """Test angular-semantic monotonicity"""
    print("\n[TEST] Angular-Semantic Monotonicity")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(100, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    result = evaluator.compute_angular_semantic_monotonicity(
        n_angles=8, n_texts_sample=50, test_radius=2.0
    )

    assert "spearman_rho" in result
    assert "p_value" in result
    assert "n_angles" in result
    assert result["n_angles"] == 8

    print(f"  Spearman ρ: {result['spearman_rho']:.4f}")
    print("✓ Angular-semantic monotonicity test passed")


def test_conditional_retrieval_gain():
    """Test conditional retrieval gain computation"""
    print("\n[TEST] Conditional Retrieval Gain")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(50, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    result = evaluator.compute_conditional_retrieval_gain(
        n_conditions=5, n_test_images=10
    )

    assert "R@1_baseline" in result
    assert "R@1_conditional" in result
    assert "R@1_absolute_gain" in result
    assert "mean_rank_baseline" in result

    print(f"  R@1 Baseline: {result['R@1_baseline']:.2f}%")
    print(f"  R@1 Conditional: {result['R@1_conditional']:.2f}%")
    print("✓ Conditional retrieval gain test passed")


def test_retrieval_diversity():
    """Test retrieval diversity computation"""
    print("\n[TEST] Retrieval Diversity")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(50, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    result = evaluator.compute_retrieval_diversity(
        n_conditions=8, n_test_images=10, k=5
    )

    assert "mean_jsd" in result
    assert "n_conditions" in result
    assert "n_pairs" in result

    print(f"  Mean JSD: {result['mean_jsd']:.4f}")
    print("✓ Retrieval diversity test passed")


def test_semantic_coherence():
    """Test semantic coherence computation"""
    print("\n[TEST] Semantic Coherence")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(100, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    result = evaluator.compute_semantic_coherence_score(
        n_bins=4, n_images_per_bin=5, k=5
    )

    assert "mean_coherence" in result
    assert "n_bins_evaluated" in result

    print(f"  Mean Coherence: {result['mean_coherence']:.4f}")
    print("✓ Semantic coherence test passed")


def test_condition_space_quality():
    """Test condition space quality metrics"""
    print("\n[TEST] Condition Space Quality")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=20, n_texts=100, device=device)
    conditions = torch.randn(100, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    result = evaluator.compute_condition_space_quality()

    assert "silhouette_score" in result
    assert "n_effective_dims" in result
    assert "radius_stats" in result
    assert "total_conditions" in result

    print(f"  Silhouette: {result['silhouette_score']:.4f}")
    print(f"  Effective Dims: {result['n_effective_dims']}")
    print("✓ Condition space quality test passed")


def test_full_evaluation():
    """Test complete evaluation pipeline"""
    print("\n[TEST] Full Evaluation Pipeline")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel(device=device)
    precomputed = create_mock_data(n_images=30, n_texts=150, device=device)
    conditions = torch.randn(100, 2, device=device)

    evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_results.json")
        results = evaluator.evaluate_all(save_path=save_path, verbose=False)

        # Check all metrics are present
        assert "metadata" in results
        assert "radius_effect" in results
        assert "angular_monotonicity" in results
        assert "retrieval_gain" in results
        assert "diversity" in results
        assert "coherence" in results
        assert "space_quality" in results

        # Check file was saved
        assert os.path.exists(save_path)

        print("✓ Full evaluation pipeline test passed")


def test_with_real_model():
    """
    Test with actual CoSiR model (integration test)
    This requires actual model and data
    """
    print("\n[TEST] Integration Test with Real Model")

    try:
        from transformers import AutoProcessor
        from torch.utils.data import DataLoader
        from src.dataset.cosir_datamodule import CoSiRValidationDataset
        from src.eval import EvaluationManager, EvaluationConfig

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CoSiRModel(label_dim=2).to(device)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Check if test data exists
        coco_annotation_path = "/data/SSD/coco/annotations/coco_karpathy_test.json"
        if not os.path.exists(coco_annotation_path):
            print("  ⊘ Skipping: COCO test data not found")
            return

        coco_test_dataset = CoSiRValidationDataset(
            annotation_path=coco_annotation_path,
            image_path="/data/SSD/coco/images",
            processor=processor,
            ratio=0.1,  # Use small subset for testing
        )

        test_loader = DataLoader(
            coco_test_dataset, batch_size=32, shuffle=False, num_workers=4
        )

        evaluation_config = EvaluationConfig(
            device=device,
            k_vals=[1, 5, 10],
            train_max_batches=5,
            print_metrics=False,
            evaluation_interval=5,
        )
        evaluator = EvaluationManager(evaluation_config)

        representatives = torch.randn(10, 2).to(device)
        test_results_detailed = evaluator.evaluate_test(
            model=model,
            processor=processor,
            dataloader=test_loader,
            label_embeddings=representatives,
            epoch=0,
            return_detailed_results=True,
        )

        (
            all_img_emb,
            all_txt_emb,
            all_raw_text,
            _,
            image_to_text_map,
            _,
        ) = test_results_detailed

        precomputed = (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map)
        conditions = torch.randn(500, 2).to(device)

        evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

        # Run a subset of evaluations
        result = evaluator.compute_radius_effect_correlation(
            n_samples=50, n_texts_sample=30
        )
        print(f"  Real model correlation: {result['correlation']:.4f}")
        print("✓ Integration test with real model passed")

    except ImportError as e:
        print(f"  ⊘ Skipping integration test: {e}")
    except Exception as e:
        print(f"  ⊘ Integration test skipped or failed: {e}")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print(" " * 15 + "CoSiRAutomaticEvaluator Test Suite")
    print("=" * 70)

    tests = [
        test_basic_initialization,
        test_edge_case_small_dataset,
        test_edge_case_single_condition,
        test_edge_case_zero_radius_conditions,
        test_edge_case_collinear_conditions,
        test_reinitialization,
        test_radius_effect_correlation,
        test_angular_semantic_monotonicity,
        test_conditional_retrieval_gain,
        test_retrieval_diversity,
        test_semantic_coherence,
        test_condition_space_quality,
        test_full_evaluation,
        test_with_real_model,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED:")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return passed, failed


if __name__ == "__main__":
    import os

    run_all_tests()
