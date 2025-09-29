#!/usr/bin/env python3
"""
Comprehensive test script for clustering.py classes and methods.
Tests both UMAP_vis and Clustering classes with various scenarios.
"""

import torch
import numpy as np
import random
from sklearn.datasets import make_blobs
from src.model.clustering import UMAP_vis, Clustering


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_synthetic_data(n_samples=1000, n_features=128, n_centers=5):
    """Create synthetic clustering data with known ground truth."""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        random_state=42,
        cluster_std=2.0
    )
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def test_umap_vis_class():
    """Test UMAP_vis class functionality."""
    print("="*60)
    print("Testing UMAP_vis Class")
    print("="*60)
    
    # Test initialization
    print("\n1. Testing initialization...")
    umap_vis = UMAP_vis(device="cuda")
    assert umap_vis.device.type == "cuda" if torch.cuda.is_available() else "cpu"
    assert umap_vis.local_model is None
    assert not umap_vis._cluster_initialized
    print("✓ Initialization successful")
    
    # Create test data
    data, labels = create_synthetic_data(n_samples=500, n_features=64, n_centers=3)
    data = data.cuda() if torch.cuda.is_available() else data
    
    # Test learn_umap with different n_components
    print("\n2. Testing learn_umap...")
    for n_comp in [2, 3, 5]:
        umap_features = umap_vis.learn_umap(data, n_components=n_comp)
        assert umap_features.shape == (500, n_comp), f"Expected shape (500, {n_comp}), got {umap_features.shape}"
        assert umap_features.device == data.device, "Device mismatch"
        assert umap_features.dtype == data.dtype, "Dtype mismatch"
        print(f"✓ UMAP with {n_comp} components: {umap_features.shape}")
    
    # Test model caching
    print("\n3. Testing model caching...")
    assert umap_vis.local_model is not None, "Model should be cached after learning"
    print("✓ Model caching works")
    
    # Test predict_umap with torch tensor
    print("\n4. Testing predict_umap with torch tensor...")
    new_data = torch.randn(100, 64, device=data.device, dtype=data.dtype)
    predicted_features = umap_vis.predict_umap(new_data)
    assert predicted_features.shape == (100, 5), f"Expected (100, 5), got {predicted_features.shape}"  # Last n_comp was 5
    assert predicted_features.device == data.device, "Device mismatch"
    print("✓ Prediction with torch tensor successful")
    
    # Test predict_umap with numpy array
    print("\n5. Testing predict_umap with numpy array...")
    new_data_np = np.random.randn(50, 64).astype(np.float32)
    predicted_features_np = umap_vis.predict_umap(new_data_np)
    assert predicted_features_np.shape == (50, 5), f"Expected (50, 5), got {predicted_features_np.shape}"
    print("✓ Prediction with numpy array successful")
    
    # Test error handling
    print("\n6. Testing error handling...")
    umap_vis_empty = UMAP_vis()
    try:
        umap_vis_empty.predict_umap(new_data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "UMAP model has not been trained yet" in str(e)
        print("✓ Error handling works correctly")
    
    # Test cluster cleanup
    print("\n7. Testing cluster cleanup...")
    umap_vis.close_cluster()
    assert not umap_vis._cluster_initialized, "Cluster should be closed"
    assert umap_vis.local_model is None, "Model should be cleared"
    print("✓ Cluster cleanup successful")


def test_clustering_class():
    """Test Clustering class functionality."""
    print("="*60)
    print("Testing Clustering Class")
    print("="*60)
    
    # Test initialization
    print("\n1. Testing initialization...")
    clustering = Clustering(device="cuda")
    assert clustering.device.type == "cuda" if torch.cuda.is_available() else "cpu"
    assert not clustering._cluster_initialized
    print("✓ Initialization successful")
    
    # Create test data
    data, true_labels = create_synthetic_data(n_samples=1000, n_features=128, n_centers=5)
    data = data.cuda() if torch.cuda.is_available() else data
    
    print(f"Test data shape: {data.shape}")
    print(f"True clusters: {len(torch.unique(true_labels))}")
    
    # Test get_umap
    print("\n2. Testing get_umap...")
    umap_features = clustering.get_umap(data, n_components=2)
    assert umap_features.shape == (1000, 2), f"Expected (1000, 2), got {umap_features.shape}"
    assert umap_features.device == data.device, "Device mismatch"
    print(f"✓ UMAP features: {umap_features.shape}")
    
    # Test get_and_predict_umap
    print("\n3. Testing get_and_predict_umap...")
    new_data = torch.randn(200, 128, device=data.device, dtype=data.dtype)
    umap_orig, umap_new = clustering.get_and_predict_umap(data, new_data, n_components=3)
    assert umap_orig.shape == (1000, 3), f"Expected (1000, 3), got {umap_orig.shape}"
    assert umap_new.shape == (200, 3), f"Expected (200, 3), got {umap_new.shape}"
    print(f"✓ UMAP with prediction: orig {umap_orig.shape}, new {umap_new.shape}")
    
    # Test get_and_predict_umap without new data
    umap_only, umap_none = clustering.get_and_predict_umap(data, None, n_components=2)
    assert umap_only.shape == (1000, 2)
    assert umap_none is None
    print("✓ UMAP without prediction works")
    
    # Use 2D UMAP for clustering tests
    umap_2d = clustering.get_umap(data, n_components=2)
    
    # Test KMeans clustering
    print("\n4. Testing KMeans clustering...")
    kmeans_labels, kmeans_centers = clustering.get_kmeans(umap_2d, n_clusters=5)
    assert kmeans_labels.shape == (1000,), f"Expected (1000,), got {kmeans_labels.shape}"
    assert kmeans_centers.shape == (5, 2), f"Expected (5, 2), got {kmeans_centers.shape}"
    assert kmeans_labels.dtype == torch.long, f"Expected torch.long, got {kmeans_labels.dtype}"
    print(f"✓ KMeans: labels {kmeans_labels.shape}, centers {kmeans_centers.shape}")
    print(f"  Unique labels: {torch.unique(kmeans_labels).tolist()}")
    
    # Test KMeans model caching
    print("\n5. Testing KMeans model caching...")
    # Same n_clusters should reuse model
    kmeans_labels2, _ = clustering.get_kmeans(umap_2d, n_clusters=5)
    assert torch.equal(kmeans_labels, kmeans_labels2), "Results should be identical with same parameters"
    
    # Different n_clusters should create new model
    kmeans_labels3, centers3 = clustering.get_kmeans(umap_2d, n_clusters=3)
    assert centers3.shape == (3, 2), f"Expected (3, 2), got {centers3.shape}"
    print("✓ KMeans model caching works")
    
    # Test kmeans_update
    print("\n6. Testing kmeans_update...")
    # Hard update
    updated_hard = clustering.kmeans_update(kmeans_labels, data, update_type="hard")
    assert updated_hard.shape == data.shape, f"Expected {data.shape}, got {updated_hard.shape}"
    
    # Soft update
    updated_soft = clustering.kmeans_update(kmeans_labels, data, update_type="soft", alpha=0.3)
    assert updated_soft.shape == data.shape, f"Expected {data.shape}, got {updated_soft.shape}"
    
    # Check that updates are different
    assert not torch.equal(updated_hard, updated_soft), "Hard and soft updates should differ"
    print("✓ KMeans updates: hard and soft modes work")
    
    # Test HDBSCAN clustering
    print("\n7. Testing HDBSCAN clustering...")
    hdb_labels, hdb_centers = clustering.get_hdbscan(
        umap_2d, min_cluster_size=50, min_sample=25, method="eom"
    )
    assert hdb_labels.shape == (1000,), f"Expected (1000,), got {hdb_labels.shape}"
    assert hdb_centers is None, "HDBSCAN should return None for centers"
    assert hdb_labels.dtype == torch.long, f"Expected torch.long, got {hdb_labels.dtype}"
    
    unique_hdb = torch.unique(hdb_labels)
    n_clusters_hdb = len(unique_hdb[unique_hdb != -1])  # Exclude noise (-1)
    noise_points = (hdb_labels == -1).sum().item()
    print(f"✓ HDBSCAN: {n_clusters_hdb} clusters, {noise_points} noise points")
    
    # Test hdbscan_update
    print("\n8. Testing hdbscan_update...")
    # Test center_only mode
    _, centers_only, counts_only = clustering.hdbscan_update(
        hdb_labels, data, center_only=True
    )
    expected_centers = n_clusters_hdb
    assert centers_only.shape[0] == expected_centers, f"Expected {expected_centers} centers"
    assert len(counts_only) == expected_centers, f"Expected {expected_centers} counts"
    print(f"✓ HDBSCAN center_only: {centers_only.shape[0]} centers")
    
    # Test hard update with noise ignore
    updated_hdb_hard, centers_hard, counts_hard = clustering.hdbscan_update(
        hdb_labels, data, update_type="hard", update_noise="ignore"
    )
    assert updated_hdb_hard.shape == data.shape
    print("✓ HDBSCAN hard update (ignore noise)")
    
    # Test soft update with noise assignment
    updated_hdb_soft, centers_soft, counts_soft = clustering.hdbscan_update(
        hdb_labels, data, update_type="soft", alpha=0.4, update_noise="assign"
    )
    assert updated_hdb_soft.shape == data.shape
    print("✓ HDBSCAN soft update (assign noise)")
    
    # Test edge case: no clusters
    print("\n9. Testing edge cases...")
    noise_labels = torch.full((100,), -1, device=data.device)  # All noise
    small_data = data[:100]
    result_noise, centers_noise, counts_noise = clustering.hdbscan_update(
        noise_labels, small_data, update_type="hard"
    )
    assert result_noise.shape == small_data.shape
    assert centers_noise.shape == (1, data.shape[1])  # Should create dummy center
    print("✓ Edge case (all noise) handled correctly")
    
    # Test cluster persistence
    print("\n10. Testing cluster persistence...")
    assert clustering._cluster_initialized, "Cluster should remain initialized"
    
    # Test cleanup
    clustering.close_cluster()
    assert not clustering._cluster_initialized, "Cluster should be closed"
    print("✓ Cluster cleanup successful")


def test_performance_comparison():
    """Test performance of optimized vs unoptimized operations."""
    print("="*60)
    print("Performance Test")
    print("="*60)
    
    import time
    
    # Create larger dataset for performance testing
    data, _ = create_synthetic_data(n_samples=2000, n_features=256, n_centers=8)
    data = data.cuda() if torch.cuda.is_available() else data
    
    clustering = Clustering()
    
    print(f"Testing with data shape: {data.shape}")
    
    # Test UMAP performance
    start_time = time.time()
    umap_features = clustering.get_umap(data, n_components=2)
    umap_time = time.time() - start_time
    print(f"UMAP time: {umap_time:.3f}s")
    
    # Test KMeans performance
    start_time = time.time()
    labels, centers = clustering.get_kmeans(umap_features, n_clusters=8)
    kmeans_time = time.time() - start_time
    print(f"KMeans time: {kmeans_time:.3f}s")
    
    # Test vectorized update performance
    start_time = time.time()
    updated = clustering.kmeans_update(labels, data, update_type="hard")
    update_time = time.time() - start_time
    print(f"Vectorized update time: {update_time:.3f}s")
    
    # Test multiple operations with persistent cluster
    start_time = time.time()
    for _ in range(3):
        umap_feat = clustering.get_umap(data, n_components=2)
        labels, _ = clustering.get_kmeans(umap_feat, n_clusters=8)
        updated = clustering.kmeans_update(labels, data)
    multi_op_time = time.time() - start_time
    print(f"3x full pipeline time: {multi_op_time:.3f}s")
    
    clustering.close_cluster()
    print("✓ Performance test completed")


def test_memory_efficiency():
    """Test memory efficiency of operations."""
    print("="*60)
    print("Memory Efficiency Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    # Test with different data sizes
    sizes = [500, 1000, 2000]
    clustering = Clustering()
    
    for size in sizes:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        data, _ = create_synthetic_data(n_samples=size, n_features=128, n_centers=5)
        data = data.cuda()
        
        # Full pipeline
        umap_features = clustering.get_umap(data, n_components=2)
        labels, centers = clustering.get_kmeans(umap_features, n_clusters=5)
        updated = clustering.kmeans_update(labels, data)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**2  # MB
        
        print(f"Size {size}: {memory_used:.1f} MB peak memory")
        torch.cuda.reset_peak_memory_stats()
    
    clustering.close_cluster()
    print("✓ Memory efficiency test completed")


def main():
    """Run all tests."""
    print("Starting Comprehensive Clustering Tests")
    print("=" * 80)
    
    set_seed(42)
    
    try:
        # Core functionality tests
        test_umap_vis_class()
        test_clustering_class()
        
        # Performance and efficiency tests
        test_performance_comparison()
        test_memory_efficiency()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)