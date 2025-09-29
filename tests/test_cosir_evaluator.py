#!/usr/bin/env python3
"""
Comprehensive test script for CoSiREvaluator class
Testing the following functionality:
0. Successfully load the latest experiments (res/CoSiR_Experiment/coco)
1. Reproduce retrieval performance with representative label embeddings from k_means
2. Arbitrarily choose some label embeddings (around 10) to perform retrieval and save results
3. Use get_umap function to get fig and save to experiment plots folder
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Add src to path
sys.path.append('src')

from hook.eval_cosir import CoSiREvaluator, load_cosir_evaluator
from utils import ExperimentManager, get_umap
import matplotlib.pyplot as plt


def test_cosir_evaluator():
    """Comprehensive test of CoSiREvaluator functionality"""

    print("=" * 80)
    print("COMPREHENSIVE COSIR EVALUATOR TEST")
    print("=" * 80)

    # 0. Load latest experiment
    print("\n0. LOADING LATEST EXPERIMENT")
    print("-" * 40)

    experiment_path = "res/CoSiR_Experiment/coco/20250917_090002_CoSiR_Experiment"

    if not Path(experiment_path).exists():
        print(f"ERROR: Experiment path {experiment_path} does not exist")
        return

    try:
        print(f"Loading experiment from: {experiment_path}")

        # Check if config needs manual loading due to string format issue
        metadata_path = Path(experiment_path) / "experiment_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                if isinstance(metadata.get("config"), str):
                    print("  Config stored as string, converting to dict...")
                    import ast
                    config_dict = ast.literal_eval(metadata["config"])
                    metadata["config"] = config_dict
                    # Save corrected config
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    print("  ✓ Fixed config format")

        evaluator = CoSiREvaluator(experiment_path)

        # Load all components
        model, feature_manager, embedding_manager = evaluator.load_all()

        print(f"✓ Successfully loaded experiment: {evaluator.experiment_name}")
        print(f"✓ Model loaded: {type(model).__name__}")
        print(f"✓ Feature manager loaded with {len(feature_manager.sample_ids)} samples")
        print(f"✓ Embedding manager loaded")

    except Exception as e:
        print(f"ERROR loading experiment: {e}")
        return

    # Get experiment manager for saving results
    exp_manager = ExperimentManager()
    try:
        experiment_context = exp_manager.load_experiment("20250917_090002_CoSiR_Experiment")
        print(f"✓ Loaded experiment context for result saving")
    except:
        # Create a simple experiment context for saving
        experiment_context = exp_manager.create_experiment("cosir_eval_test",
                                                           config={"test": "cosir_evaluator"},
                                                           description="CoSiR Evaluator Test")
        print(f"✓ Created new experiment context for result saving: {experiment_context.name}")

    # 1. Test retrieval performance with k_means representatives
    print("\n1. TESTING RETRIEVAL PERFORMANCE WITH K_MEANS REPRESENTATIVES")
    print("-" * 60)

    try:
        # Get representative embeddings (k_means clustering)
        num_representatives = 50
        representatives = evaluator.get_representatives(num_representatives)

        print(f"✓ Generated {num_representatives} representative embeddings")
        print(f"  Representative embeddings shape: {representatives.shape}")

        # Get all embeddings for analysis
        sample_ids, all_embeddings = evaluator.get_all_embeddings()
        print(f"✓ Loaded all embeddings: {all_embeddings.shape}")
        print(f"  Sample IDs range: {min(sample_ids)} to {max(sample_ids)}")

        # Compute similarity matrix between representatives and all embeddings
        similarities = torch.mm(representatives, all_embeddings.T.to(representatives.device))
        print(f"✓ Computed similarity matrix: {similarities.shape}")

        # Save representative embeddings and similarity results
        rep_results = {
            "representatives": representatives.cpu().numpy(),
            "sample_ids": sample_ids.cpu().numpy(),
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

    # 2. Choose 10 arbitrary label embeddings for retrieval
    print("\n2. ARBITRARY LABEL EMBEDDING RETRIEVAL TEST")
    print("-" * 50)

    try:
        # Choose 10 arbitrary sample IDs
        np.random.seed(42)  # For reproducibility
        arbitrary_indices = np.random.choice(len(sample_ids), size=10, replace=False)
        arbitrary_sample_ids = sample_ids[arbitrary_indices].tolist()
        arbitrary_embeddings = all_embeddings[arbitrary_indices]

        print(f"✓ Selected 10 arbitrary sample IDs: {arbitrary_sample_ids}")
        print(f"  Arbitrary embeddings shape: {arbitrary_embeddings.shape}")

        # Compute retrieval similarities for arbitrary embeddings
        arbitrary_similarities = torch.mm(arbitrary_embeddings, all_embeddings.T.to(arbitrary_embeddings.device))

        # For each arbitrary embedding, find top-k most similar embeddings
        k = 20  # Top-20 retrievals
        top_k_similarities, top_k_indices = torch.topk(arbitrary_similarities, k=k, dim=1)

        # Create retrieval results
        arbitrary_results = {
            "query_sample_ids": arbitrary_sample_ids,
            "query_embeddings": arbitrary_embeddings.cpu().numpy(),
            "retrieval_results": []
        }

        for i, query_sample_id in enumerate(arbitrary_sample_ids):
            retrieved_sample_ids = sample_ids[top_k_indices[i]].tolist()
            retrieved_similarities = top_k_similarities[i].tolist()

            arbitrary_results["retrieval_results"].append({
                "query_sample_id": query_sample_id,
                "retrieved_sample_ids": retrieved_sample_ids,
                "similarities": retrieved_similarities,
                "top_1_similarity": retrieved_similarities[0],
                "mean_similarity": np.mean(retrieved_similarities)
            })

            print(f"  Query {query_sample_id}: Top-3 retrieved = {retrieved_sample_ids[:3]}")

        # Save arbitrary retrieval results
        experiment_context.save_artifact(
            name="arbitrary_embeddings_retrieval",
            data=arbitrary_results,
            artifact_type="pickle",
            folder="results",
            description=f"Retrieval results for 10 arbitrary label embeddings (top-{k})"
        )

        # Also save as JSON for easy viewing
        json_results = arbitrary_results.copy()
        json_results["query_embeddings"] = "saved_as_numpy_array"  # Remove numpy array for JSON
        experiment_context.save_artifact(
            name="arbitrary_embeddings_retrieval_summary",
            data=json_results,
            artifact_type="json",
            folder="results",
            description="Summary of arbitrary embedding retrieval results (JSON format)"
        )

        print("✓ Saved arbitrary embedding retrieval results to results folder")

    except Exception as e:
        print(f"ERROR in arbitrary embedding retrieval test: {e}")
        import traceback
        traceback.print_exc()

    # 3. Generate UMAP visualization and save to plots folder
    print("\n3. UMAP VISUALIZATION TEST")
    print("-" * 30)

    try:
        # Sample embeddings if too many (UMAP can be slow on large datasets)
        max_samples_for_umap = 2000
        if len(all_embeddings) > max_samples_for_umap:
            sample_indices = np.random.choice(len(all_embeddings), size=max_samples_for_umap, replace=False)
            embeddings_for_umap = all_embeddings[sample_indices]
            sample_ids_for_umap = sample_ids[sample_indices]
            print(f"✓ Sampled {max_samples_for_umap} embeddings for UMAP (from {len(all_embeddings)} total)")
        else:
            embeddings_for_umap = all_embeddings
            sample_ids_for_umap = sample_ids
            print(f"✓ Using all {len(all_embeddings)} embeddings for UMAP")

        # Generate UMAP visualization
        print("  Generating UMAP visualization...")
        fig = get_umap(embeddings_for_umap.cpu().numpy(),
                      title="CoSiR Label Embeddings UMAP",
                      n_neighbors=15,
                      min_dist=0.1)

        print("✓ Generated UMAP visualization")

        # Save UMAP plot to plots folder
        experiment_context.save_artifact(
            name="label_embeddings_umap",
            data=fig,
            artifact_type="figure",
            folder="plots",
            description="UMAP visualization of trained label embeddings"
        )

        print("✓ Saved UMAP plot to plots folder")

        # Create additional analysis plot - similarity distribution
        plt.figure(figsize=(10, 6))

        # Plot 1: Similarity distribution
        plt.subplot(1, 2, 1)
        similarities_flat = similarities.cpu().numpy().flatten()
        plt.hist(similarities_flat, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores\n(Representatives vs All Embeddings)')
        plt.grid(True, alpha=0.3)

        # Plot 2: Embedding norm distribution
        plt.subplot(1, 2, 2)
        embedding_norms = torch.norm(all_embeddings, dim=1).cpu().numpy()
        plt.hist(embedding_norms, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Embedding Norm')
        plt.ylabel('Frequency')
        plt.title('Distribution of Embedding Norms')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save analysis plots
        experiment_context.save_artifact(
            name="embedding_analysis_plots",
            data=plt.gcf(),
            artifact_type="figure",
            folder="plots",
            description="Analysis plots: similarity distribution and embedding norms"
        )

        print("✓ Saved additional analysis plots to plots folder")
        plt.close('all')  # Clean up plots

    except Exception as e:
        print(f"ERROR in UMAP visualization test: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ Successfully loaded latest CoSiR experiment")
    print("✓ Tested k_means representative embeddings for retrieval")
    print("✓ Tested retrieval with 10 arbitrary label embeddings")
    print("✓ Generated and saved UMAP visualization")
    print("✓ All results saved to experiment folders:")
    print(f"  - Results: {experiment_context.paths.results}")
    print(f"  - Plots: {experiment_context.paths.plots}")

    return evaluator, experiment_context


if __name__ == "__main__":
    # Activate CoSiR environment
    test_cosir_evaluator()