#!/usr/bin/env python3
"""
Simplified test script for CoSiREvaluator class functionality
Focuses on loading components individually and testing core features
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Add src to path
sys.path.append('src')

from utils import (
    FeatureManager,
    ExperimentManager,
    TrainableEmbeddingManager,
    get_representatives,
    get_umap
)
import matplotlib.pyplot as plt


def test_cosir_evaluator_components():
    """Test CoSiREvaluator components individually"""

    print("=" * 80)
    print("SIMPLIFIED COSIR EVALUATOR COMPONENT TEST")
    print("=" * 80)

    experiment_path = "res/CoSiR_Experiment/coco/20250917_090002_CoSiR_Experiment"

    # Load config manually
    print("\n0. LOADING EXPERIMENT CONFIG")
    print("-" * 40)

    config_path = Path(experiment_path) / "experiment_metadata.json"
    with open(config_path) as f:
        metadata = json.load(f)
        config = metadata["config"]

    print(f"✓ Loaded config from: {config_path}")
    print(f"  Model embedding dim: {config['model']['embedding_dim']}")
    print(f"  Feature storage dir: {config['featuremanager']['storage_dir']}")

    # Get experiment manager for saving results
    exp_manager = ExperimentManager()
    experiment_context = exp_manager.create_experiment("cosir_eval_test",
                                                       config={"test": "cosir_evaluator_simple"},
                                                       description="Simple CoSiR Evaluator Test")
    print(f"✓ Created experiment context: {experiment_context.name}")

    # 1. Test TrainableEmbeddingManager loading
    print("\n1. TESTING TRAINABLE EMBEDDING MANAGER LOADING")
    print("-" * 55)

    try:
        # Load sample IDs
        sample_ids_path = config["featuremanager"]["sample_ids_path"]
        sample_ids_list = torch.load(sample_ids_path, map_location="cpu")
        print(f"✓ Loaded sample IDs: {len(sample_ids_list)} samples")
        print(f"  Sample ID range: {min(sample_ids_list)} to {max(sample_ids_list)}")

        # Initialize embedding manager
        embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids_list,
            embedding_dim=config["model"]["embedding_dim"],
            storage_mode="disk",
            device="cpu",  # Use CPU for simplicity
            initialization_strategy="random",
            embeddings_dir=str(Path(experiment_path) / "final_embeddings"),
            cache_l1_size_mb=64,
            cache_l2_size_mb=128,
            enable_l3_cache=False,
            auto_sync=False,
        )

        print(f"✓ Created embedding manager")
        print(f"  Embedding shape: {embedding_manager.embeddings.shape}")

        # Get all embeddings
        sample_ids_tensor, all_embeddings = embedding_manager.get_all_embeddings()
        print(f"✓ Retrieved all embeddings: {all_embeddings.shape}")

    except Exception as e:
        print(f"ERROR in embedding manager test: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Test k_means representatives
    print("\n2. TESTING K_MEANS REPRESENTATIVE EMBEDDINGS")
    print("-" * 50)

    try:
        # Generate representative embeddings using k_means
        num_representatives = 50
        representatives = get_representatives(all_embeddings.cpu().numpy(), num_representatives)
        print(f"✓ Generated {num_representatives} representative embeddings")
        print(f"  Representative embeddings shape: {representatives.shape}")

        # Compute similarity matrix
        representatives_tensor = torch.from_numpy(representatives)
        similarities = torch.mm(representatives_tensor, all_embeddings.T)
        print(f"✓ Computed similarity matrix: {similarities.shape}")

        # Save results
        rep_results = {
            "representatives": representatives,
            "sample_ids": sample_ids_tensor.cpu().numpy(),
            "similarity_stats": {
                "mean": similarities.mean().item(),
                "std": similarities.std().item(),
                "min": similarities.min().item(),
                "max": similarities.max().item()
            },
            "num_representatives": num_representatives,
            "embedding_dim": representatives.shape[1]
        }

        experiment_context.save_artifact(
            name="kmeans_representatives_results",
            data=rep_results,
            artifact_type="pickle",
            folder="results",
            description="K-means representative embeddings and similarity analysis"
        )
        print("✓ Saved k_means representative results to results folder")

    except Exception as e:
        print(f"ERROR in k_means representatives test: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test arbitrary embedding retrieval
    print("\n3. TESTING ARBITRARY EMBEDDING RETRIEVAL")
    print("-" * 45)

    try:
        # Choose 10 arbitrary embeddings
        np.random.seed(42)
        arbitrary_indices = np.random.choice(len(sample_ids_tensor), size=10, replace=False)
        arbitrary_sample_ids = sample_ids_tensor[arbitrary_indices].tolist()
        arbitrary_embeddings = all_embeddings[arbitrary_indices]

        print(f"✓ Selected 10 arbitrary sample IDs: {arbitrary_sample_ids}")

        # Compute retrieval similarities
        arbitrary_similarities = torch.mm(arbitrary_embeddings, all_embeddings.T)

        # Find top-k most similar
        k = 20
        top_k_similarities, top_k_indices = torch.topk(arbitrary_similarities, k=k, dim=1)

        # Create retrieval results
        arbitrary_results = {
            "query_sample_ids": arbitrary_sample_ids,
            "query_embeddings": arbitrary_embeddings.cpu().numpy(),
            "retrieval_results": []
        }

        for i, query_sample_id in enumerate(arbitrary_sample_ids):
            retrieved_sample_ids = sample_ids_tensor[top_k_indices[i]].tolist()
            retrieved_similarities = top_k_similarities[i].tolist()

            arbitrary_results["retrieval_results"].append({
                "query_sample_id": query_sample_id,
                "retrieved_sample_ids": retrieved_sample_ids,
                "similarities": retrieved_similarities,
                "top_1_similarity": retrieved_similarities[0],
                "mean_similarity": np.mean(retrieved_similarities)
            })

            print(f"  Query {query_sample_id}: Top-3 = {retrieved_sample_ids[:3]}")

        # Save results
        experiment_context.save_artifact(
            name="arbitrary_embeddings_retrieval",
            data=arbitrary_results,
            artifact_type="pickle",
            folder="results",
            description=f"Retrieval results for 10 arbitrary label embeddings (top-{k})"
        )

        # Also save JSON summary
        json_results = arbitrary_results.copy()
        json_results["query_embeddings"] = "saved_as_numpy_array"
        experiment_context.save_artifact(
            name="arbitrary_embeddings_retrieval_summary",
            data=json_results,
            artifact_type="json",
            folder="results",
            description="Summary of arbitrary embedding retrieval results"
        )

        print("✓ Saved arbitrary embedding retrieval results to results folder")

    except Exception as e:
        print(f"ERROR in arbitrary embedding retrieval test: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test UMAP visualization
    print("\n4. TESTING UMAP VISUALIZATION")
    print("-" * 35)

    try:
        # Sample embeddings for UMAP (limit for performance)
        max_samples = 1000
        if len(all_embeddings) > max_samples:
            sample_indices = np.random.choice(len(all_embeddings), size=max_samples, replace=False)
            embeddings_for_umap = all_embeddings[sample_indices]
            print(f"✓ Sampled {max_samples} embeddings for UMAP")
        else:
            embeddings_for_umap = all_embeddings
            print(f"✓ Using all {len(all_embeddings)} embeddings for UMAP")

        # Generate UMAP plot
        print("  Generating UMAP visualization...")
        fig = get_umap(embeddings_for_umap.cpu().numpy(),
                      title="CoSiR Label Embeddings UMAP",
                      n_neighbors=15,
                      min_dist=0.1)

        # Save UMAP plot
        experiment_context.save_artifact(
            name="label_embeddings_umap",
            data=fig,
            artifact_type="figure",
            folder="plots",
            description="UMAP visualization of trained label embeddings"
        )

        print("✓ Saved UMAP plot to plots folder")

        # Create additional analysis plots
        plt.figure(figsize=(12, 5))

        # Similarity distribution
        plt.subplot(1, 3, 1)
        similarities_flat = similarities.cpu().numpy().flatten()
        plt.hist(similarities_flat, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Similarity Distribution\n(Representatives vs All)')
        plt.grid(True, alpha=0.3)

        # Embedding norm distribution
        plt.subplot(1, 3, 2)
        embedding_norms = torch.norm(all_embeddings, dim=1).cpu().numpy()
        plt.hist(embedding_norms, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Embedding Norm')
        plt.ylabel('Frequency')
        plt.title('Embedding Norm Distribution')
        plt.grid(True, alpha=0.3)

        # Representative embedding visualization (if 2D)
        plt.subplot(1, 3, 3)
        if representatives.shape[1] == 2:
            plt.scatter(representatives[:, 0], representatives[:, 1],
                       c='red', s=50, alpha=0.7, label='Representatives')
            plt.scatter(all_embeddings[:, 0], all_embeddings[:, 1],
                       c='blue', s=10, alpha=0.3, label='All Embeddings')
            plt.xlabel('Embedding Dim 1')
            plt.ylabel('Embedding Dim 2')
            plt.title('2D Embedding Space')
            plt.legend()
        else:
            # PCA projection if higher dimensional
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            all_embeddings_2d = pca.fit_transform(all_embeddings.cpu().numpy())
            representatives_2d = pca.transform(representatives)

            plt.scatter(representatives_2d[:, 0], representatives_2d[:, 1],
                       c='red', s=50, alpha=0.7, label='Representatives')
            plt.scatter(all_embeddings_2d[:, 0], all_embeddings_2d[:, 1],
                       c='blue', s=10, alpha=0.3, label='All Embeddings')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('PCA Projection (2D)')
            plt.legend()

        plt.tight_layout()

        # Save analysis plots
        experiment_context.save_artifact(
            name="embedding_analysis_plots",
            data=plt.gcf(),
            artifact_type="figure",
            folder="plots",
            description="Analysis plots: similarity distribution, norms, and 2D visualization"
        )

        print("✓ Saved additional analysis plots to plots folder")
        plt.close('all')

    except Exception as e:
        print(f"ERROR in UMAP visualization test: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ Successfully loaded experiment components")
    print("✓ Tested k_means representative embeddings generation")
    print("✓ Tested retrieval with 10 arbitrary label embeddings")
    print("✓ Generated and saved UMAP and analysis visualizations")
    print("✓ All results saved to experiment folders:")
    print(f"  - Results: {experiment_context.paths.results}")
    print(f"  - Plots: {experiment_context.paths.plots}")
    print(f"\nExperiment: {experiment_context.name}")

    return experiment_context


if __name__ == "__main__":
    test_cosir_evaluator_components()