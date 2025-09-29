#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
sys.path.append('/project/CoSiR/src')

from utils.embedding_manager import TrainableEmbeddingManager

def benchmark_embedding_operations(
    sample_ids: List[int],
    embedding_dim: int = 128,
    chunk_size: int = 100,
    storage_mode: str = "memory",
    auto_sync: bool = True,
    num_operations: int = 50,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Benchmark embedding operations with different configurations
    
    Returns:
        Dict with timing results for different operations
    """
    
    print(f"\n=== Benchmarking Configuration ===")
    print(f"Sample IDs: {len(sample_ids)} samples")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Chunk size: {chunk_size}")
    print(f"Storage mode: {storage_mode}")
    print(f"Auto sync: {auto_sync}")
    print(f"Operations: {num_operations}")
    print(f"Batch size: {batch_size}")
    
    # Create temporary directory for this test
    temp_dir = tempfile.mkdtemp(prefix=f"embed_test_{storage_mode}_{chunk_size}_")
    
    try:
        # Initialize manager
        start_init = time.time()
        manager = TrainableEmbeddingManager(
            sample_ids=sample_ids,
            embedding_dim=embedding_dim,
            storage_mode=storage_mode,
            embeddings_dir=temp_dir,
            chunk_size=chunk_size,
            auto_sync=auto_sync
        )
        init_time = time.time() - start_init
        
        # Test get_embeddings performance
        get_times = []
        update_times = []
        
        print(f"Running {num_operations} operations...")
        
        for i in tqdm(range(num_operations), desc="Testing operations"):
            # Random batch of sample IDs
            batch_sample_ids = np.random.choice(sample_ids, size=min(batch_size, len(sample_ids)), replace=False).tolist()
            
            # Test get_embeddings timing
            start_get = time.time()
            embeddings = manager.get_embeddings(batch_sample_ids)
            get_time = time.time() - start_get
            get_times.append(get_time)
            
            # Test update_embeddings timing
            new_embeddings = torch.randn(len(batch_sample_ids), embedding_dim)
            start_update = time.time()
            manager.update_embeddings(batch_sample_ids, new_embeddings)
            update_time = time.time() - start_update
            update_times.append(update_time)
        
        # Get final disk usage info
        disk_info = manager.get_disk_usage_info()
        
        return {
            'init_time': init_time,
            'get_mean': np.mean(get_times),
            'get_std': np.std(get_times),
            'get_total': np.sum(get_times),
            'update_mean': np.mean(update_times),
            'update_std': np.std(update_times),
            'update_total': np.sum(update_times),
            'total_time': np.sum(get_times) + np.sum(update_times),
            'disk_usage_mb': disk_info['total_size_mb'],
            'num_chunks': disk_info['num_chunks'],
            'temp_dir': temp_dir
        }
        
    except Exception as e:
        print(f"Error in benchmark: {e}")
        return {'error': str(e), 'temp_dir': temp_dir}
    finally:
        # Note: We'll keep temp directories for analysis, clean up manually later
        pass

def run_comprehensive_benchmark():
    """
    Run comprehensive benchmarks across different configurations
    """
    
    # Test configurations
    sample_counts = [100, 500, 1000, 2000]  # Different dataset sizes
    chunk_sizes = [50, 100, 200, 500, 1000]  # Different chunk sizes
    storage_modes = ["memory", "disk"]
    auto_sync_options = [True, False]
    
    results = []
    
    print("=== Running Comprehensive Embedding Performance Benchmark ===")
    
    for sample_count in sample_counts:
        sample_ids = list(range(sample_count))
        
        for chunk_size in chunk_sizes:
            # Skip chunk sizes larger than sample count
            if chunk_size > sample_count:
                continue
                
            for storage_mode in storage_modes:
                for auto_sync in auto_sync_options:
                    config = {
                        'sample_count': sample_count,
                        'chunk_size': chunk_size,
                        'storage_mode': storage_mode,
                        'auto_sync': auto_sync
                    }
                    
                    print(f"\n--- Testing: {config} ---")
                    
                    benchmark_result = benchmark_embedding_operations(
                        sample_ids=sample_ids,
                        chunk_size=chunk_size,
                        storage_mode=storage_mode,
                        auto_sync=auto_sync,
                        num_operations=30,  # Reduced for comprehensive testing
                        batch_size=32
                    )
                    
                    # Combine config with results
                    result = {**config, **benchmark_result}
                    results.append(result)
                    
                    # Print key metrics
                    if 'error' not in result:
                        print(f"  Init: {result['init_time']:.4f}s")
                        print(f"  Get (avg): {result['get_mean']:.4f}s ± {result['get_std']:.4f}s")
                        print(f"  Update (avg): {result['update_mean']:.4f}s ± {result['update_std']:.4f}s")
                        print(f"  Total: {result['total_time']:.4f}s")
                        print(f"  Disk: {result['disk_usage_mb']:.2f}MB ({result['num_chunks']} chunks)")
    
    return results

def analyze_results(results: List[Dict[str, Any]]):
    """
    Analyze benchmark results and identify performance issues
    """
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Filter out error results
    df_clean = df[~df['error'].notna()].copy() if 'error' in df.columns else df.copy()
    
    if df_clean.empty:
        print("No valid results to analyze!")
        return
    
    print("\n=== Performance Analysis ===")
    
    # Analysis 1: Impact of chunk size
    print("\n1. Impact of Chunk Size on Performance:")
    chunk_analysis = df_clean.groupby(['chunk_size', 'storage_mode']).agg({
        'get_mean': 'mean',
        'update_mean': 'mean',
        'total_time': 'mean',
        'init_time': 'mean'
    }).round(4)
    print(chunk_analysis)
    
    # Analysis 2: Impact of storage mode
    print("\n2. Impact of Storage Mode:")
    storage_analysis = df_clean.groupby(['storage_mode', 'auto_sync']).agg({
        'get_mean': 'mean',
        'update_mean': 'mean',
        'total_time': 'mean',
        'init_time': 'mean'
    }).round(4)
    print(storage_analysis)
    
    # Analysis 3: Impact of auto_sync
    print("\n3. Impact of Auto-Sync:")
    sync_analysis = df_clean.groupby(['auto_sync', 'storage_mode']).agg({
        'update_mean': 'mean',
        'total_time': 'mean'
    }).round(4)
    print(sync_analysis)
    
    # Analysis 4: Scale with sample count
    print("\n4. Scaling with Sample Count:")
    scale_analysis = df_clean.groupby('sample_count').agg({
        'get_mean': 'mean',
        'update_mean': 'mean',
        'init_time': 'mean'
    }).round(4)
    print(scale_analysis)
    
    # Identify performance issues
    print("\n=== Performance Issues Identified ===")
    
    # Find configurations with worst performance
    worst_get = df_clean.loc[df_clean['get_mean'].idxmax()]
    worst_update = df_clean.loc[df_clean['update_mean'].idxmax()]
    worst_total = df_clean.loc[df_clean['total_time'].idxmax()]
    
    print(f"\nWorst GET performance:")
    print(f"  Config: chunk_size={worst_get['chunk_size']}, storage_mode={worst_get['storage_mode']}, auto_sync={worst_get['auto_sync']}")
    print(f"  Time: {worst_get['get_mean']:.4f}s ± {worst_get['get_std']:.4f}s")
    
    print(f"\nWorst UPDATE performance:")
    print(f"  Config: chunk_size={worst_update['chunk_size']}, storage_mode={worst_update['storage_mode']}, auto_sync={worst_update['auto_sync']}")
    print(f"  Time: {worst_update['update_mean']:.4f}s ± {worst_update['update_std']:.4f}s")
    
    print(f"\nWorst TOTAL performance:")
    print(f"  Config: chunk_size={worst_total['chunk_size']}, storage_mode={worst_total['storage_mode']}, auto_sync={worst_total['auto_sync']}")
    print(f"  Time: {worst_total['total_time']:.4f}s")
    
    # Check for concerning patterns
    print("\n=== Performance Patterns ===")
    
    # Check if performance degrades with larger chunk sizes (in disk mode)
    disk_results = df_clean[df_clean['storage_mode'] == 'disk'].copy()
    if not disk_results.empty:
        chunk_correlation = disk_results['chunk_size'].corr(disk_results['total_time'])
        print(f"Correlation between chunk_size and total_time (disk mode): {chunk_correlation:.4f}")
        
        if chunk_correlation > 0.5:
            print("  WARNING: Performance significantly degrades with larger chunk sizes in disk mode!")
        
    # Check auto_sync impact
    if 'auto_sync' in df_clean.columns:
        sync_on = df_clean[df_clean['auto_sync'] == True]['update_mean'].mean()
        sync_off = df_clean[df_clean['auto_sync'] == False]['update_mean'].mean()
        sync_ratio = sync_on / sync_off if sync_off > 0 else float('inf')
        print(f"Auto-sync performance ratio: {sync_ratio:.2f}x slower when enabled")
        
        if sync_ratio > 2.0:
            print("  WARNING: Auto-sync causes significant performance degradation!")
    
    return df_clean

def create_performance_plots(df: pd.DataFrame, save_dir: str):
    """
    Create performance visualization plots
    """
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Plot 1: Chunk size vs performance
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    for storage_mode in df['storage_mode'].unique():
        subset = df[df['storage_mode'] == storage_mode]
        chunk_perf = subset.groupby('chunk_size')['get_mean'].mean()
        plt.plot(chunk_perf.index, chunk_perf.values, marker='o', label=f'{storage_mode} mode')
    plt.xlabel('Chunk Size')
    plt.ylabel('Avg Get Time (s)')
    plt.title('Get Performance vs Chunk Size')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 3, 2)
    for storage_mode in df['storage_mode'].unique():
        subset = df[df['storage_mode'] == storage_mode]
        chunk_perf = subset.groupby('chunk_size')['update_mean'].mean()
        plt.plot(chunk_perf.index, chunk_perf.values, marker='o', label=f'{storage_mode} mode')
    plt.xlabel('Chunk Size')
    plt.ylabel('Avg Update Time (s)')
    plt.title('Update Performance vs Chunk Size')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 3, 3)
    for storage_mode in df['storage_mode'].unique():
        subset = df[df['storage_mode'] == storage_mode]
        chunk_perf = subset.groupby('chunk_size')['init_time'].mean()
        plt.plot(chunk_perf.index, chunk_perf.values, marker='o', label=f'{storage_mode} mode')
    plt.xlabel('Chunk Size')
    plt.ylabel('Init Time (s)')
    plt.title('Initialization Time vs Chunk Size')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Storage mode comparison
    plt.subplot(2, 3, 4)
    storage_comparison = df.groupby('storage_mode')[['get_mean', 'update_mean']].mean()
    storage_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Storage Mode Performance Comparison')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Plot 3: Auto-sync impact
    plt.subplot(2, 3, 5)
    sync_comparison = df.groupby('auto_sync')[['get_mean', 'update_mean']].mean()
    sync_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Auto-Sync Impact on Performance')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Plot 4: Scaling with sample count
    plt.subplot(2, 3, 6)
    scale_data = df.groupby('sample_count')[['get_mean', 'update_mean', 'init_time']].mean()
    scale_data.plot(ax=plt.gca())
    plt.title('Performance Scaling with Sample Count')
    plt.xlabel('Sample Count')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path / 'embedding_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def investigate_disk_mode_issues():
    """
    Deep dive into disk mode performance issues
    """
    
    print("\n=== Deep Dive: Disk Mode Performance Issues ===")
    
    sample_ids = list(range(500))  # Medium dataset
    chunk_sizes = [50, 100, 200, 500]
    
    detailed_results = []
    
    for chunk_size in chunk_sizes:
        print(f"\nInvestigating chunk_size = {chunk_size}")
        
        temp_dir = tempfile.mkdtemp(prefix=f"disk_investigation_{chunk_size}_")
        
        try:
            # Create manager
            manager = TrainableEmbeddingManager(
                sample_ids=sample_ids,
                embedding_dim=128,
                storage_mode="disk",
                embeddings_dir=temp_dir,
                chunk_size=chunk_size,
                auto_sync=True
            )
            
            # Test operations with detailed timing
            batch_sample_ids = sample_ids[:32]  # Fixed batch
            
            # Detailed get_embeddings analysis
            get_times = []
            chunk_loads = []
            
            for i in range(10):
                # Track chunk loading
                initial_loaded_chunks = len(manager.loaded_chunks)
                
                start_time = time.time()
                embeddings = manager.get_embeddings(batch_sample_ids)
                get_time = time.time() - start_time
                
                final_loaded_chunks = len(manager.loaded_chunks)
                chunks_loaded = final_loaded_chunks - initial_loaded_chunks
                
                get_times.append(get_time)
                chunk_loads.append(chunks_loaded)
                
                print(f"  Operation {i+1}: {get_time:.4f}s, chunks loaded: {chunks_loaded}")
            
            # Detailed update_embeddings analysis  
            update_times = []
            chunk_syncs = []
            
            for i in range(10):
                new_embeddings = torch.randn(len(batch_sample_ids), 128)
                
                start_time = time.time()
                manager.update_embeddings(batch_sample_ids, new_embeddings)
                update_time = time.time() - start_time
                
                dirty_chunks = len(manager.dirty_chunks)
                
                update_times.append(update_time)
                chunk_syncs.append(dirty_chunks)
                
                print(f"  Update {i+1}: {update_time:.4f}s, dirty chunks: {dirty_chunks}")
            
            disk_info = manager.get_disk_usage_info()
            
            detailed_results.append({
                'chunk_size': chunk_size,
                'get_mean': np.mean(get_times),
                'get_std': np.std(get_times),
                'update_mean': np.mean(update_times),
                'update_std': np.std(update_times),
                'avg_chunks_loaded': np.mean(chunk_loads),
                'avg_dirty_chunks': np.mean(chunk_syncs),
                'total_chunks': disk_info['num_chunks'],
                'disk_usage_mb': disk_info['total_size_mb'],
                'temp_dir': temp_dir
            })
            
        except Exception as e:
            print(f"Error with chunk_size {chunk_size}: {e}")
            detailed_results.append({
                'chunk_size': chunk_size,
                'error': str(e),
                'temp_dir': temp_dir
            })
    
    return detailed_results

def main():
    """
    Main debugging function
    """
    
    print("=== Embedding Manager Performance Debugging ===")
    print("This script investigates performance issues with TrainableEmbeddingManager")
    print("focusing on chunk size, storage mode, and auto_sync impacts.\n")
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    # Analyze results
    df = analyze_results(results)
    
    # Create performance plots
    debug_dir = "/project/CoSiR/src/test/20250911_embedding_performance_debugging"
    if df is not None and not df.empty:
        create_performance_plots(df, debug_dir)
    
    # Deep dive into disk mode issues
    detailed_results = investigate_disk_mode_issues()
    
    # Save detailed results
    results_file = Path(debug_dir) / "performance_results.csv"
    if df is not None and not df.empty:
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    # Save detailed disk investigation
    detailed_file = Path(debug_dir) / "detailed_disk_investigation.csv"
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Detailed disk investigation saved to: {detailed_file}")
    
    # Create summary report
    create_summary_report(df, detailed_df, debug_dir)
    
    print(f"\n=== Debugging Complete ===")
    print(f"All results and plots saved to: {debug_dir}")

def create_summary_report(df: pd.DataFrame, detailed_df: pd.DataFrame, save_dir: str):
    """
    Create a summary report of findings
    """
    
    report_path = Path(save_dir) / "20250911_embedding_performance_log.md"
    
    with open(report_path, 'w') as f:
        f.write("# Embedding Manager Performance Investigation\n\n")
        f.write("**Date:** 2025-09-11\n")
        f.write("**Problem:** Training gets slower with larger chunk sizes and disk storage mode\n\n")
        
        f.write("## Investigation Summary\n\n")
        f.write("### Key Findings\n\n")
        
        if df is not None and not df.empty:
            # Performance by storage mode
            storage_perf = df.groupby('storage_mode')[['get_mean', 'update_mean']].mean()
            f.write("**Storage Mode Performance:**\n")
            f.write(f"- Memory mode: Get {storage_perf.loc['memory', 'get_mean']:.4f}s, Update {storage_perf.loc['memory', 'update_mean']:.4f}s\n")
            if 'disk' in storage_perf.index:
                f.write(f"- Disk mode: Get {storage_perf.loc['disk', 'get_mean']:.4f}s, Update {storage_perf.loc['disk', 'update_mean']:.4f}s\n")
            f.write("\n")
            
            # Chunk size impact
            chunk_correlation = df[df['storage_mode'] == 'disk']['chunk_size'].corr(df[df['storage_mode'] == 'disk']['total_time'])
            f.write(f"**Chunk Size Impact (Disk Mode):**\n")
            f.write(f"- Correlation with total time: {chunk_correlation:.4f}\n")
            if chunk_correlation > 0.5:
                f.write("- **WARNING:** Performance significantly degrades with larger chunks!\n")
            f.write("\n")
            
            # Auto-sync impact
            if len(df['auto_sync'].unique()) > 1:
                sync_on = df[df['auto_sync'] == True]['update_mean'].mean()
                sync_off = df[df['auto_sync'] == False]['update_mean'].mean()
                f.write(f"**Auto-Sync Impact:**\n")
                f.write(f"- With auto-sync: {sync_on:.4f}s\n")
                f.write(f"- Without auto-sync: {sync_off:.4f}s\n")
                f.write(f"- Performance ratio: {sync_on/sync_off:.2f}x\n\n")
        
        f.write("### Root Cause Analysis\n\n")
        f.write("Based on the TrainableEmbeddingManager implementation review:\n\n")
        
        f.write("1. **Inefficient Chunk Loading in Disk Mode:**\n")
        f.write("   - `_load_embeddings_from_disk()` loads entire chunks for each batch\n")
        f.write("   - No intelligent caching strategy like FeatureManager\n")
        f.write("   - Larger chunks = more unnecessary data loaded per operation\n\n")
        
        f.write("2. **Redundant Disk I/O with Auto-Sync:**\n")
        f.write("   - `_sync_updated_samples_to_disk()` triggers on every update\n")
        f.write("   - Loads, modifies, and saves entire chunks repeatedly\n")
        f.write("   - No batching of sync operations\n\n")
        
        f.write("3. **Cache Management Issues:**\n")
        f.write("   - `loaded_chunks` dictionary grows without intelligent eviction\n")
        f.write("   - No LRU or size-based cache management\n")
        f.write("   - Memory usage can grow unbounded\n\n")
        
        f.write("### Recommended Optimizations\n\n")
        f.write("1. **Implement Intelligent Caching:**\n")
        f.write("   - Add LRU cache with size limits\n")
        f.write("   - Pre-load adjacent chunks based on access patterns\n")
        f.write("   - Share caching strategy with FeatureManager\n\n")
        
        f.write("2. **Batch Sync Operations:**\n")
        f.write("   - Accumulate dirty chunks and sync in batches\n")
        f.write("   - Add configurable sync intervals\n")
        f.write("   - Implement async sync for non-critical updates\n\n")
        
        f.write("3. **Optimize Chunk Size Selection:**\n")
        f.write("   - Align chunk boundaries with typical batch sizes\n")
        f.write("   - Consider embedding access patterns in training\n")
        f.write("   - Add automatic chunk size optimization\n\n")
        
        f.write("4. **Memory Management:**\n")
        f.write("   - Implement chunk cache eviction policies\n")
        f.write("   - Add memory pressure monitoring\n")
        f.write("   - Provide cache statistics and tuning\n\n")
    
    print(f"Summary report saved to: {report_path}")

if __name__ == "__main__":
    main()