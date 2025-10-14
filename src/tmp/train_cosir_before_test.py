from bdb import Breakpoint
import os
import torch
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoProcessor
from torch.optim.lr_scheduler import CosineAnnealingLR


from src.dataset import (
    CoSiRTrainingChunkDataset,
    CoSiRValidationDataset,
    FeatureExtractionDataset,
)
from src.model import CoSiRModel
from src.utils import (
    FeatureManager,
    ExperimentManager,
    TrainableEmbeddingManager,
    replace_with_most_different,
)
from src.metrics import LabelContrastiveLoss


def train_cosir(*args, **kwargs):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "epochs": 50,
        "lr": 1e-6,
        "num_workers": 4,  # Reduced to prevent memory issues
        "batch_size": 256,  # Reduced from 512 to prevent OOM
        "embedding_dim": 2,
        "load_existing_features": True,
        "chunk_size": 1024,  # Reduced from 1024 to prevent memory pressure
        "cache_size_mb": 512,  # Reduced to leave room for embedding cache
        "use_compression": False,
        "train_annotation_path": "/data/SSD/coco/annotations/coco_karpathy_train.json",
        "test_annotation_path": "/data/SSD/coco/annotations/coco_karpathy_test.json",
        "train_image_path": "/data/SSD/coco/images",
        "test_image_path": "/data/SSD/coco/images",
        "experiment_name": "comprehensive_cosir_test",
        "tags": ["deep_clustering", "multimodal", "comprehensive_test"],
        # ========== NEW: Optimized Embedding Caching Configuration ==========
        "embedding_caching": {
            # Multi-tier cache configuration (optimized for system memory)
            "cache_l1_size_mb": 128,  # Reduced to prevent OOM
            "cache_l2_size_mb": 256,  # Reduced to prevent OOM
            "enable_l3_cache": True,  # Enable disk cache
            # Sync optimization settings
            "auto_sync": False,  # Enable automatic sync
            "sync_batch_size": 10,  # Batch sync every N updates (reduces I/O)
            # Performance optimization
            "embedding_chunk_size": None,  # Auto-optimize based on batch_size (batch_size * 4)
            # Monitoring and debugging
            "enable_performance_monitoring": True,  # Track cache performance
            "log_cache_stats_every_n_epochs": 5,  # Log cache stats every 5 epochs
            "force_sync_at_epoch_end": True,  # Ensure all changes saved per epoch
        },
    }

    feature_config = {
        "storage_dir": "/project/CoSiR/data/comprehensive_cosir_test/features",
        "sample_ids_path": "/project/CoSiR/data/comprehensive_cosir_test/sample_ids.pt",
        "primary_backend": "chunked",
        "chunked_storage": {
            "enabled": True,
            "chunk_size": config["chunk_size"],
            "compression": config["use_compression"],
        },
        "cache": {
            "l1_size_mb": config["cache_size_mb"],
            "l2_size_mb": config["cache_size_mb"] * 2,
            "l3_path": None,
        },
    }

    # Initialize model
    print("Initializing model")
    model = CoSiRModel(label_dim=config["embedding_dim"]).to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Setup criteria and optimizer and scheduler
    print("Initializing criteria and optimizer and scheduler")
    criteria = LabelContrastiveLoss(
        margin=0.3,
        lambda_pos=1.0,
        lambda_neg=1.0,
        lambda_labelchange=0.1,
        lambda_preserve=0.1,
        return_dict=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=5e-2,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=1e-9,
        last_epoch=-1,
    )

    # Initialize pre-extraction dataset
    print("Initializing pre-extraction dataset")
    pre_extraction_dataset = FeatureExtractionDataset(
        annotation_path=config["train_annotation_path"],
        image_path=config["train_image_path"],
        processor=processor,
        ratio=1,
    )
    pre_extraction_dataloader = DataLoader(
        pre_extraction_dataset,
        batch_size=config["chunk_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # Check either load existing sample ids
    if (
        os.path.exists(feature_config["sample_ids_path"])
        and config["load_existing_features"]
    ):
        feature_manager = FeatureManager(
            feature_config["storage_dir"], config=feature_config
        )
        print("Loading existing sample ids")
        sample_ids_list = torch.load(feature_config["sample_ids_path"])
        feature_manager.load_features()
        print(f"Loaded {len(sample_ids_list)} sample ids")
    else:
        print("Creating new sample ids")
        feature_manager = FeatureManager(
            features_dir=feature_config["storage_dir"],
            chunk_size=config["chunk_size"],
            config=feature_config,
        )
        # Create sample ids list
        sample_ids_list = []

        # Pre-extract features
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(pre_extraction_dataloader)):
                image_inputs, text_inputs, sample_ids = batch
                sample_ids = [int(id) for id in sample_ids]
                image_inputs = image_inputs.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                chunk_img, chunk_txt, _, txt_full = model.encode_img_txt(
                    image_inputs, text_inputs
                )

                chunk_img, chunk_txt, txt_full = (
                    chunk_img.detach().cpu(),
                    chunk_txt.detach().cpu(),
                    txt_full.detach().cpu(),
                )

                # print(chunk_img.shape, chunk_txt.shape, txt_full.shape)

                # Add features to feature manager
                feature_manager.add_features_chunk(
                    batch_id,
                    chunk_img,
                    chunk_txt,
                    txt_full,
                    sample_ids,
                )

                # Add sample ids to list
                sample_ids_list.extend(sample_ids)

        # Save sample ids
        torch.save(sample_ids_list, feature_config["sample_ids_path"])

    # Initialize folder manager
    exp_manager = ExperimentManager("res/comprehensive_cosir_test/experiments")

    # Create comprehensive experiment
    experiment = exp_manager.create_experiment(
        name=config["experiment_name"],
        config=config,
        tags=config["tags"],
        description="Comprehensive test of integrated feature and experiment management systems",
    )

    print(f"Created experiment: {experiment.name}")
    print(f"Experiment directory: {experiment.directory}")

    # ========== OPTIMIZED: TrainableEmbeddingManager with Intelligent Caching ==========
    embedding_config = config["embedding_caching"]

    # Auto-optimize chunk size based on batch size if not specified
    embedding_chunk_size = embedding_config["embedding_chunk_size"]
    if embedding_chunk_size is None:
        # Optimal chunk size: 4x batch size, capped at reasonable limits
        embedding_chunk_size = min(
            max(config["batch_size"] * 4, 100), config["chunk_size"]
        )
        print(
            f"Auto-optimized embedding chunk size: {embedding_chunk_size} (4x batch_size)"
        )

    print("Initializing optimized TrainableEmbeddingManager...")
    print(f"  L1 cache: {embedding_config['cache_l1_size_mb']}MB")
    print(f"  L2 cache: {embedding_config['cache_l2_size_mb']}MB")
    print(
        f"  L3 cache: {'enabled' if embedding_config['enable_l3_cache'] else 'disabled'}"
    )
    print(f"  Sync batch size: {embedding_config['sync_batch_size']}")
    print(f"  Chunk size: {embedding_chunk_size}")

    embedding_manager = TrainableEmbeddingManager(
        sample_ids=sample_ids_list,  # Use actual sample IDs from feature extraction
        embedding_dim=config["embedding_dim"],
        storage_mode="disk",
        embeddings_dir=str(experiment.directory / "training_embeddings"),
        # Optimized caching parameters
        cache_l1_size_mb=embedding_config["cache_l1_size_mb"],
        cache_l2_size_mb=embedding_config["cache_l2_size_mb"],
        enable_l3_cache=embedding_config["enable_l3_cache"],
        # Batched sync optimization
        auto_sync=embedding_config["auto_sync"],
        sync_batch_size=embedding_config["sync_batch_size"],
        # Chunk size optimization
        chunk_size=embedding_chunk_size,
    )

    # Optimize cache settings based on actual batch size
    embedding_manager.optimize_cache_settings(config["batch_size"])
    print("TrainableEmbeddingManager optimization complete!")

    # ========== CHUNK-BASED LOADING: Use CoSiRTrainingChunkDataset ==========
    # Calculate number of chunks needed
    num_samples = len(sample_ids_list)
    num_chunks = (num_samples + config["chunk_size"] - 1) // config[
        "chunk_size"
    ]  # Ceiling division
    print(
        f"Dataset: {num_samples} samples, {num_chunks} chunks of size {config['chunk_size']}"
    )

    # Use CoSiRTrainingChunkDataset for chunk-based loading
    train_set = CoSiRTrainingChunkDataset(
        feature_manager=feature_manager,
        sample_ids=sample_ids_list,  # Full sample IDs list for reference
        enable_prefetch=True,
    )

    # DataLoader with batch_size=chunk_size since we're loading chunks directly, we only use batch_idx as chunk_id
    # num_workers=0 to eliminate worker process issues completely
    train_loader = DataLoader(
        train_set,
        batch_size=config["chunk_size"],  # Load one chunk at a time (batch_idx only)
        shuffle=False,  # We'll handle chunk order manually if needed
        num_workers=0,  # No worker processes - eliminates all worker issues!
    )

    test_set = CoSiRValidationDataset(
        annotation_path=config["test_annotation_path"],
        image_path=config["test_image_path"],
        processor=processor,
        ratio=1.0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Initialize performance tracking
    embedding_performance_log = []

    # Memory monitoring function
    def log_memory_usage(stage=""):
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            print(
                f"Memory usage {stage}: {memory.percent:.1f}% ({memory.available/1024/1024/1024:.1f}GB available)"
            )

    # Check initial memory
    try:
        import psutil

        HAS_PSUTIL = True
        log_memory_usage("before training")
    except ImportError:
        HAS_PSUTIL = False
        print("psutil not available - memory monitoring disabled")

    for epoch in range(config["epochs"]):
        experiment.current_epoch = epoch

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Track epoch start time for performance monitoring
        import time

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Use batch_idx as chunk_id for direct chunk loading
            chunk_id = batch_idx

            # Load features by chunk directly
            features_data = feature_manager.get_features_by_chunk(chunk_id)

            img_features = features_data["img_features"].to(device, non_blocking=True)
            txt_features = features_data["txt_features"].to(device, non_blocking=True)
            txt_full = features_data["txt_full"].to(device, non_blocking=True)

            # Get embeddings by chunk directly from optimized manager
            chunk_sample_ids, label_embeddings_data = (
                embedding_manager.get_embeddings_by_chunk(chunk_id)
            )
            label_embeddings = torch.nn.Parameter(
                label_embeddings_data.to(device), requires_grad=True
            )

            # Create temp optimizer only for label embeddings to preserve main optimizer state
            base_lr = optimizer.param_groups[0]["lr"]
            label_optimizer = torch.optim.AdamW(
                [{"params": [label_embeddings], "lr": base_lr * 5e4}],
                weight_decay=5e-2,
            )

            comb_emb, label_embedding_proj = model.combine(
                txt_features,
                txt_full,
                label_embeddings,
                epoch=epoch,
                return_label_proj=True,
            )

            label_embedding_neg = replace_with_most_different(label_embeddings)

            comb_emb_neg = model.combine(
                txt_features,
                txt_full,
                label_embedding_neg,
                epoch=epoch,
                return_label_proj=False,
            )

            loss_dict = criteria(
                img_features,
                txt_features,
                comb_emb,
                comb_emb_neg,
                label_embeddings,
                label_embedding_proj,
            )

            loss = loss_dict["total_loss"]

            epoch_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            label_optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model parameters with accumulated state
            label_optimizer.step()  # Update label embeddings with fresh optimizer

            # Update embeddings by chunk directly using optimized batched sync
            embedding_manager.update_embeddings_by_chunk(
                chunk_id, chunk_sample_ids, label_embeddings
            )

        # ========== EPOCH END: Performance Monitoring & Sync ==========
        epoch_time = time.time() - epoch_start_time

        # Force sync all chunks at epoch end (ensures data safety)
        if embedding_config["force_sync_at_epoch_end"]:
            embedding_manager.force_sync_all_chunks()

        # Log epoch performance
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

        # Cache performance monitoring
        if embedding_config["enable_performance_monitoring"]:
            cache_performance = embedding_manager.get_cache_performance()
            embedding_performance_log.append(
                {
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "epoch_time": epoch_time,
                    "cache_hit_rate": cache_performance["cache_hit_rate"],
                    "total_requests": cache_performance["total_requests"],
                    "disk_load_ratio": cache_performance["disk_load_ratio"],
                    "sync_operations": cache_performance["sync_operations"],
                    "l1_usage_percent": cache_performance["l1_usage_percent"],
                    "l2_usage_percent": cache_performance["l2_usage_percent"],
                }
            )

            # Log detailed cache stats every N epochs
            if (epoch + 1) % embedding_config["log_cache_stats_every_n_epochs"] == 0:
                print(f"\\n=== Embedding Cache Performance (Epoch {epoch}) ===")
                print(f"  Cache hit rate: {cache_performance['cache_hit_rate']:.1%}")
                print(f"  Total cache requests: {cache_performance['total_requests']}")
                print(f"  Disk load ratio: {cache_performance['disk_load_ratio']:.1%}")
                print(f"  Sync operations: {cache_performance['sync_operations']}")
                print(f"  L1 cache usage: {cache_performance['l1_usage_percent']:.1f}%")
                print(f"  L2 cache usage: {cache_performance['l2_usage_percent']:.1f}%")

                # Get comprehensive disk usage info
                disk_info = embedding_manager.get_disk_usage_info()
                print(f"  Disk usage: {disk_info['total_size_mb']:.2f}MB")
                print(f"  Dirty chunks: {disk_info['dirty_chunks_count']}")
                print(f"  Pending updates: {disk_info['pending_updates_count']}")

                # Log memory usage
                log_memory_usage(f"after epoch {epoch}")
                print()

                # Adaptive cache management if memory pressure detected
                if HAS_PSUTIL:
                    memory = psutil.virtual_memory()
                    if memory.percent > 85.0:  # High memory usage
                        print(
                            "⚠️  High memory usage detected - triggering adaptive cache eviction"
                        )
                        embedding_manager.cache.adaptive_eviction()
                        log_memory_usage("after cache eviction")

        scheduler.step()

    # ========== TRAINING COMPLETE: Final Performance Summary ==========
    # Save final embeddings and performance log
    embedding_manager.save_final_embeddings(
        str(experiment.directory / "final_embeddings")
    )

    # Save performance log for analysis
    if embedding_config["enable_performance_monitoring"] and embedding_performance_log:
        import json

        performance_log_path = experiment.directory / "embedding_performance_log.json"
        with open(performance_log_path, "w") as f:
            json.dump(embedding_performance_log, f, indent=2)
        print(f"Embedding performance log saved to: {performance_log_path}")

        # Print training performance summary
        print("\\n" + "=" * 60)
        print("EMBEDDING CACHE PERFORMANCE SUMMARY")
        print("=" * 60)

        final_cache_perf = embedding_manager.get_cache_performance()
        final_disk_info = embedding_manager.get_disk_usage_info()

        avg_cache_hit_rate = sum(
            log["cache_hit_rate"] for log in embedding_performance_log
        ) / len(embedding_performance_log)
        total_sync_ops = final_cache_perf["sync_operations"]
        total_requests = final_cache_perf["total_requests"]

        print(f"Configuration:")
        print(f"  L1 Cache: {embedding_config['cache_l1_size_mb']}MB")
        print(f"  L2 Cache: {embedding_config['cache_l2_size_mb']}MB")
        print(
            f"  L3 Cache: {'Enabled' if embedding_config['enable_l3_cache'] else 'Disabled'}"
        )
        print(f"  Sync Batch Size: {embedding_config['sync_batch_size']}")
        print(f"  Chunk Size: {embedding_chunk_size}")
        print()
        print(f"Performance Results:")
        print(f"  Average Cache Hit Rate: {avg_cache_hit_rate:.1%}")
        print(f"  Total Cache Requests: {total_requests:,}")
        print(f"  Total Sync Operations: {total_sync_ops}")
        print(f"  Final Disk Usage: {final_disk_info['total_size_mb']:.2f}MB")
        print(f"  Final L1 Usage: {final_cache_perf['l1_usage_percent']:.1f}%")
        print(f"  Final L2 Usage: {final_cache_perf['l2_usage_percent']:.1f}%")

        # Performance improvement estimates
        estimated_old_syncs = (
            len(embedding_performance_log) * num_batches
        )  # Old: sync per batch
        sync_reduction = (
            (estimated_old_syncs - total_sync_ops) / estimated_old_syncs * 100
        )

        print()
        print(f"Optimization Impact:")
        print(f"  Estimated sync reduction: {sync_reduction:.1f}%")
        print(f"  ({estimated_old_syncs:,} → {total_sync_ops} sync operations)")
        print("=" * 60)

    print("Training Complete! 🎉")


if __name__ == "__main__":
    train_cosir()
