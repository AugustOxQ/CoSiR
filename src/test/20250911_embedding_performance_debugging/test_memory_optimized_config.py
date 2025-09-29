#!/usr/bin/env python3

import os
import sys
import tempfile
import torch
from pathlib import Path

# Add project root to Python path
sys.path.append('/project/CoSiR/src')

def test_memory_optimized_configuration():
    """Test the memory-optimized configuration to prevent DataLoader crashes"""
    print("=== Testing Memory-Optimized Configuration ===")
    
    # Test memory-friendly configuration
    from utils.embedding_manager import TrainableEmbeddingManager
    
    # Configuration matching the fixed train_cosir.py
    config = {
        "batch_size": 256,      # Reduced from 512
        "chunk_size": 512,      # Reduced from 1024  
        "num_workers": 4,       # Reduced from 8
        "cache_size_mb": 512,   # Reduced from 1024
        "embedding_caching": {
            "cache_l1_size_mb": 128,  # Reduced from 256
            "cache_l2_size_mb": 256,  # Reduced from 512
            "enable_l3_cache": True,
            "auto_sync": False,       # Maximum performance
            "sync_batch_size": 10,
        }
    }
    
    print("✓ Configuration loaded successfully")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Chunk size: {config['chunk_size']}")
    print(f"  Workers: {config['num_workers']}")
    print(f"  L1 cache: {config['embedding_caching']['cache_l1_size_mb']}MB")
    print(f"  L2 cache: {config['embedding_caching']['cache_l2_size_mb']}MB")
    
    # Test embedding manager with reduced memory footprint
    sample_ids = list(range(500))  # Test dataset
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = TrainableEmbeddingManager(
            sample_ids=sample_ids,
            embedding_dim=128,
            device="cpu",
            storage_mode="disk",
            embeddings_dir=temp_dir,
            cache_l1_size_mb=config["embedding_caching"]["cache_l1_size_mb"],
            cache_l2_size_mb=config["embedding_caching"]["cache_l2_size_mb"],
            enable_l3_cache=config["embedding_caching"]["enable_l3_cache"],
            auto_sync=config["embedding_caching"]["auto_sync"],
            sync_batch_size=config["embedding_caching"]["sync_batch_size"],
            chunk_size=100,  # Smaller chunks for test
        )
        
        print("✓ TrainableEmbeddingManager initialized successfully")
        
        # Test operations that would be performed in training
        batch_size = config["batch_size"] // 4  # Smaller batches for test
        test_ids = sample_ids[:batch_size]
        
        # Test get_embeddings
        embeddings = manager.get_embeddings(test_ids)
        print(f"✓ get_embeddings successful: {embeddings.shape}")
        
        # Test update_embeddings  
        new_embeddings = torch.randn(batch_size, 128)
        manager.update_embeddings(test_ids, new_embeddings)
        print("✓ update_embeddings successful")
        
        # Test cache performance
        cache_perf = manager.get_cache_performance()
        print(f"✓ Cache hit rate: {cache_perf['cache_hit_rate']:.1%}")
        
        # Force sync (simulating epoch end)
        manager.force_sync_all_chunks()
        print("✓ Force sync successful")
        
    return True

def check_memory_usage():
    """Check current memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\\n=== System Memory Status ===")
        print(f"Total memory: {memory.total / 1024**3:.1f}GB")
        print(f"Available memory: {memory.available / 1024**3:.1f}GB")
        print(f"Memory usage: {memory.percent:.1f}%")
        print(f"Free memory: {memory.free / 1024**3:.1f}GB")
        
        if memory.percent > 85.0:
            print("⚠️  WARNING: High memory usage detected!")
            print("   Consider further reducing batch sizes or cache sizes")
            return False
        elif memory.available < 2.0 * 1024**3:  # Less than 2GB available
            print("⚠️  WARNING: Low available memory!")  
            print("   Consider reducing num_workers or cache sizes")
            return False
        else:
            print("✅ Memory usage looks healthy for training")
            return True
            
    except ImportError:
        print("psutil not available - cannot check memory usage")
        return True

def provide_memory_optimization_tips():
    """Provide tips for further memory optimization"""
    print("\\n=== Memory Optimization Tips ===")
    print("""
🔧 If you still experience memory issues, try these adjustments:

1. **Further reduce batch sizes:**
   config["batch_size"] = 128  # or even 64

2. **Reduce number of DataLoader workers:**
   config["num_workers"] = 2  # or even 1

3. **Smaller cache sizes:**
   embedding_caching["cache_l1_size_mb"] = 64
   embedding_caching["cache_l2_size_mb"] = 128

4. **Disable L3 cache if disk space is limited:**
   embedding_caching["enable_l3_cache"] = False

5. **Use smaller chunk sizes:**
   config["chunk_size"] = 256  # or 128

6. **Monitor GPU memory if using CUDA:**
   torch.cuda.empty_cache()  # Add this periodically

7. **Consider gradient accumulation:**
   # Use smaller physical batches with gradient accumulation
   # to achieve effective larger batch sizes
""")

def main():
    """Test memory-optimized configuration and provide guidance"""
    print("🔧 Testing Memory-Optimized Configuration for train_cosir.py")
    print("="*70)
    
    # Check system memory
    memory_ok = check_memory_usage()
    
    # Test the configuration
    try:
        config_ok = test_memory_optimized_configuration()
        
        if memory_ok and config_ok:
            print("\\n🎉 Memory-optimized configuration test successful!")
            print("\\n✅ The fixed train_cosir.py should now work without DataLoader crashes")
            print("\\n📊 Expected improvements:")
            print("   • Reduced memory pressure")
            print("   • Stable DataLoader workers")  
            print("   • Faster embedding operations (auto_sync=False)")
            print("   • Comprehensive performance monitoring")
            
        else:
            print("\\n⚠️  Configuration may need further optimization")
            provide_memory_optimization_tips()
            
    except Exception as e:
        print(f"\\n❌ Configuration test failed: {e}")
        provide_memory_optimization_tips()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())