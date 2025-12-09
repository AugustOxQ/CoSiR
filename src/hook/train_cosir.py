import time
import os
import torch
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoProcessor
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import matplotlib.pyplot as plt

from src.dataset import (
    CoSiRTrainingChunkDataset,
    CoSiRValidationDataset,
    FeatureExtractionDataset,
    FeatureExtractionConceptualDataset,
)
from src.model import CoSiRModel, Clustering, UMAP_vis
from src.eval import EvaluationManager, EvaluationConfig
from src.utils import (
    FeatureManager,
    ExperimentManager,
    TrainableEmbeddingManager,
    replace_with_most_different,
    get_representatives,
    get_representatives_polar_grid,
    get_umap,
    visualize_ideal_condition_space,
    visualize_angular_semantics_fast,
    visualize_angular_semantics_text_to_image_fast,
    CoSiRAutomaticEvaluator,
)
from src.metrics import LabelContrastiveLoss, LabelContrastiveLoss_enhance


def train_cosir(cfg, logger):
    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    feature_config = {
        "storage_dir": cfg.featuremanager.storage_dir,
        "sample_ids_path": cfg.featuremanager.sample_ids_path,
        "primary_backend": cfg.featuremanager.primary_backend,
        "chunked_storage": {
            "enabled": cfg.featuremanager.chunked_storage.enabled,
            "chunk_size": cfg.featuremanager.chunk_size,
            "compression": cfg.featuremanager.chunked_storage.compression,
        },
        "cache": {
            "l1_size_mb": cfg.featuremanager.cache.l1_size_mb,
            "l2_size_mb": cfg.featuremanager.cache.l2_size_mb,
            "l3_path": cfg.featuremanager.cache.l3_path,
        },
    }

    evaluation_config = EvaluationConfig(
        device=device,
        k_vals=cfg.eval.k_vals,
        train_max_batches=cfg.eval.train_max_batches,
        print_metrics=cfg.eval.print_metrics,
        evaluation_interval=(
            cfg.eval.evaluation_interval if cfg.eval.evaluation_interval > 0 else 5
        ),
    )

    # Initialize model
    print("Initializing model")
    model = CoSiRModel(
        label_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.hidden_dim,
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.model.clip_model, use_fast=False)

    # Setup criteria and optimizer and scheduler
    print("Initializing criteria and optimizer and scheduler")
    criteria = LabelContrastiveLoss_enhance(
        margin=cfg.loss.margin,
        lambda_1=cfg.loss.lambda_1,
        lambda_2=cfg.loss.lambda_2,
        lambda_3=cfg.loss.lambda_3,
        lambda_4=cfg.loss.lambda_4,
        return_dict=cfg.loss.return_dict,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.T_max if cfg.scheduler.T_max > 0 else cfg.train.epochs,
        eta_min=cfg.scheduler.eta_min,
        last_epoch=-1,
    )

    # Check either load existing sample ids
    if (
        os.path.exists(feature_config["sample_ids_path"])
        and cfg.train.load_existing_features
    ):
        print("Use existing FeatureManager")
        feature_manager = FeatureManager(
            feature_config["storage_dir"], config=feature_config, preload_index=False
        )  # preload_index=False means we don't load the index mapping from the existing chunk files
        print("Loading existing sample ids")
        sample_ids_list = torch.load(feature_config["sample_ids_path"])
        print(f"Loaded {len(sample_ids_list)} sample ids")
    else:  # Optimized that only create preextraction dataset when necessary
        print("Creating new sample ids")
        feature_manager = FeatureManager(
            features_dir=feature_config["storage_dir"],
            chunk_size=cfg.featuremanager.chunk_size,
            config=feature_config,
        )
        # Create sample ids list
        sample_ids_list = []

        # Initialize pre-extraction dataset
        print("Initializing pre-extraction dataset")
        if "conceptual" in cfg.data.dataset_type:
            preextractfeatureclass = FeatureExtractionConceptualDataset
        else:
            preextractfeatureclass = FeatureExtractionDataset

        pre_extraction_dataset = preextractfeatureclass(
            annotation_path=cfg.data.train_annotation_path,
            image_path=cfg.data.train_image_path,
            processor=processor,
            ratio=1,
        )
        pre_extraction_dataloader = DataLoader(
            pre_extraction_dataset,
            batch_size=cfg.featuremanager.chunk_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )

        # Pre-extract features
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(pre_extraction_dataloader)):
                image_inputs, text_inputs, sample_ids = batch
                sample_ids = [int(id) for id in sample_ids]
                image_inputs = image_inputs.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                chunk_img, chunk_txt, _, _ = model.encode_img_txt(
                    image_inputs, text_inputs
                )  # DEBUG: txt_full is not used

                txt_full = torch.zeros_like(chunk_txt)

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
    exp_manager = ExperimentManager(cfg.experiment.results_dir)

    # Create comprehensive experiment
    experiment = exp_manager.create_experiment(
        name=cfg.experiment.name,
        config=cfg,
        tags=cfg.experiment.tags,
        description=cfg.experiment.description,
    )

    print(f"Created experiment: {experiment.name}")
    print(f"Experiment directory: {experiment.directory}")

    # Initialize evaluator
    evaluator = EvaluationManager(evaluation_config)

    # clustering = Clustering(device=device)
    umap_vis = UMAP_vis(device=device)

    # ========== OPTIMIZED: TrainableEmbeddingManager with Intelligent Caching ==========

    # Auto-optimize chunk size based on batch size if not specified
    embedding_chunk_size = cfg.embeddingmanager.embedding_chunk_size
    if embedding_chunk_size is None:
        # Optimal chunk size: 4x batch size, capped at reasonable limits
        embedding_chunk_size = cfg.featuremanager.chunk_size
        print(
            f"Auto-optimized embedding chunk size: {embedding_chunk_size} (4x batch_size)"
        )

    embedding_manager = TrainableEmbeddingManager(
        sample_ids=sample_ids_list,  # Use actual sample IDs from feature extraction
        embedding_dim=cfg.model.embedding_dim,
        storage_mode=cfg.embeddingmanager.storage_mode,
        device=device,
        initialization_strategy=cfg.embeddingmanager.initialization_strategy,
        embeddings_dir=str(experiment.directory / "training_embeddings"),
        # Optimized caching parameters
        cache_l1_size_mb=cfg.embeddingmanager.cache_l1_size_mb,
        cache_l2_size_mb=cfg.embeddingmanager.cache_l2_size_mb,
        enable_l3_cache=cfg.embeddingmanager.enable_l3_cache,
        # Batched sync optimization
        auto_sync=cfg.embeddingmanager.auto_sync,
        sync_batch_size=cfg.embeddingmanager.sync_batch_size,
        # Chunk size optimization
        chunk_size=embedding_chunk_size,
    )

    # ========== Template Embedding Initialization ==========
    if cfg.train.initialization_strategy == "imgtxt":
        # Determine template directory (one level up from directory, two levels up from embeddings_dir)
        template_dir = experiment.directory.parent / "template_embeddings"
        template_exists = template_dir.exists() and list(
            template_dir.glob("embeddings_chunk_*.pt")
        )

        # Try to load from template if it exists and is enabled
        if template_exists and getattr(cfg.train, "use_template_embeddings", True):
            print("Attempting to load from template embeddings...")
            try:
                embedding_manager.load_imgtxt_template()
            except Exception as e:
                print(f"Failed to load template: {e}")
                print("Falling back to imgtxt initialization...")
                embedding_manager.initialize_embeddings_imgtxt(
                    feature_manager, model, device
                )
        else:
            # Initialize embeddings with imgtxt strategy
            print("Initializing embeddings with imgtxt strategy...")
            embedding_manager.initialize_embeddings_imgtxt(
                feature_manager,
                model,
                device,
                factor=cfg.train.imgtxt_factor,
            )

            # Save as template embeddings if enabled (default: True)
            if getattr(cfg.train, "save_as_template_embeddings", True):
                print("Storing embeddings as template for future use...")
                embedding_manager.store_imgtxt_template()

    # Optimize cache settings based on actual batch size
    embedding_manager.optimize_cache_settings(cfg.featuremanager.chunk_size)

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
        batch_size=cfg.featuremanager.chunk_size,  # Load one chunk at a time (batch_idx only)
        shuffle=False,  # We'll handle chunk order manually if needed
        num_workers=cfg.train.num_workers,  # No worker processes - eliminates all worker issues!
    )

    test_set = CoSiRValidationDataset(
        annotation_path=cfg.data.test_annotation_path,
        image_path=cfg.data.test_image_path,
        processor=processor,
        ratio=1.0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    global_step = 0
    for epoch in range(cfg.train.epochs):
        experiment.current_epoch = epoch

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Track epoch start time for performance monitoring
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
                [
                    {
                        "params": [label_embeddings],
                        "lr": base_lr * cfg.optimizer.label_lr_multiplier,
                    }
                ],
                weight_decay=cfg.optimizer.weight_decay,
            )

            comb_emb = model.combine(
                txt_features,
                txt_full,
                label_embeddings,
                epoch=epoch,
                return_label_proj=False,
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
                model,
            )

            loss = loss_dict["total_loss"]

            epoch_loss += loss.item()
            num_batches += 1

            # Log batch-level loss components
            batch_log_dict = {
                f"train/batch_{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
            }
            batch_log_dict.update(
                {
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                    "train/step": global_step,
                }
            )
            logger.log_metrics(batch_log_dict)

            optimizer.zero_grad()
            label_optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model parameters with accumulated state
            label_optimizer.step()  # Update label embeddings with fresh optimizer
            global_step += 1

            # Update embeddings by chunk directly using optimized batched sync
            embedding_manager.update_embeddings_by_chunk(
                chunk_id, chunk_sample_ids, label_embeddings
            )

        # ========== EPOCH END: Performance Monitoring & Sync ==========
        epoch_time = time.time() - epoch_start_time

        # Force sync all chunks at epoch end (ensures data safety)
        # if cfg.embeddingmanager.force_sync_at_epoch_end:
        # embedding_manager.force_sync_all_chunks()

        # Log epoch performance
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

        # Log to wandb logger
        logger.log_metrics(
            {
                "train/epoch": epoch,
                "train/loss": avg_loss,
                "train/epoch_time": epoch_time,
            }
        )

        scheduler.step()

        # # Validation loop
        # model.eval()
        # with torch.no_grad():
        #     train_results = evaluator.evaluate_train(
        #         model=model,
        #         feature_manager=feature_manager,
        #         embedding_manager=embedding_manager,
        #         dataloader=train_loader,
        #         device=device,
        #         epoch=epoch,
        #     )

        #     # Log train results
        #     # logger.log_metrics({train_results.metrics})

        if (
            epoch == 0
            or epoch % evaluation_config.evaluation_interval == 0
            or epoch == cfg.train.epochs - 1
        ):
            model.eval()
            with torch.no_grad():
                # Test evaluation
                _, label_embeddings_all = embedding_manager.get_all_embeddings()
                representatives = get_representatives_polar_grid(
                    label_embeddings_all.cpu(),
                    (
                        cfg.train.representative_number
                        if epoch != cfg.train.epochs - 1
                        else 30  # Use higher number of representatives for the last epoch
                    ),
                )
                test_results_detailed = evaluator.evaluate_test(
                    model=model,
                    processor=processor,
                    dataloader=test_loader,
                    label_embeddings=representatives,  # Your label embeddings
                    epoch=epoch,
                    return_detailed_results=True,
                )

                (
                    all_img_emb,
                    all_txt_emb,
                    all_raw_text,
                    text_to_image_map,
                    image_to_text_map,
                    test_results,
                ) = test_results_detailed  # type: ignore

                all_raw_image = test_set.get_all_raw_image()

                # Log test results
                for metric, value in test_results.metrics.items():
                    logger.log_metrics({metric: value})

                # Visualize test results
                # # Check whether label embeddings are 2d or higher dimensional
                if label_embeddings_all.shape[1] == 2:
                    umap_features = label_embeddings_all.cpu().numpy()
                else:
                    umap_features = umap_vis.learn_umap(
                        label_embeddings_all, close_cluster=True
                    )

                # Visualize label embeddings
                fig = get_umap(
                    umap_features,
                    umap_labels=None,
                    epoch=epoch,
                    no_outlier=True,
                    samples_to_track=[0, 1, 2, 3, 4],
                )
                experiment.save_artifact(
                    name=f"label_embeddings_umap_{epoch}",
                    data=fig,
                    artifact_type="figure",
                    folder="plots",
                    description=f"UMAP visualization of trained label embeddings at epoch {epoch}",
                )

                fig2 = visualize_ideal_condition_space(umap_features, epoch)
                experiment.save_artifact(
                    name=f"ideal_condition_space_{epoch}",
                    data=fig2,
                    artifact_type="figure",
                    folder="plots",
                    description=f"Ideal condition space visualization at epoch {epoch}",
                )

                logger.log_metrics(
                    {
                        "vis/umap": wandb.Image(fig),
                        "vis/ideal_condition_space": wandb.Image(fig2),
                    }
                )

                plt.close("all")

                for tmp_round in range(10):
                    fig3 = visualize_angular_semantics_fast(
                        label_embeddings_all.cpu().numpy(),
                        model,
                        (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                        device=device,
                    )
                    experiment.save_artifact(
                        name=f"angular_semantics_fast_{epoch}_tmp_{tmp_round}",
                        data=fig3,
                        artifact_type="figure",
                        folder="plots",
                        description=f"Angular semantics visualization at epoch {epoch}",
                    )

                    fig4 = visualize_angular_semantics_text_to_image_fast(
                        label_embeddings_all.cpu().numpy(),
                        model,
                        (
                            all_img_emb,
                            all_txt_emb,
                            all_raw_image,
                            all_raw_text,
                        ),
                        device=device,
                    )

                    experiment.save_artifact(
                        name=f"angular_semantics_text_to_image_{epoch}_tmp_{tmp_round}",
                        data=fig4,
                        artifact_type="figure",
                        folder="plots",
                        description=f"Angular semantics visualization (text-to-image) at epoch {epoch}",
                    )

                    plt.close("all")

                cosir_automatic_evaluator = CoSiRAutomaticEvaluator(
                    model,
                    (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                    label_embeddings_all,
                    device,
                )
                result = cosir_automatic_evaluator.evaluate_all()
                logger.log_metrics(result)

    # ========== TRAINING COMPLETE: Final Performance Summary ==========
    # Save final embeddings and model combiner state dictionary
    embedding_manager.save_final_embeddings(
        str(experiment.directory / "final_embeddings")
    )

    # Save embedding mapping id mapping
    experiment.save_artifact(
        name="chunk_mapping",
        data=embedding_manager.chunk_mapping,
        artifact_type="pickle",
        description="Chunk mapping id mapping",
        folder="embeddings",
    )

    experiment.save_artifact(
        name="id_to_chunk_index",
        data=embedding_manager.id_to_chunk_index,
        artifact_type="pickle",
        description="Id to chunk index mapping",
        folder="embeddings",
    )

    experiment.save_artifact(
        name="final_model",
        folder="checkpoints",
        data=model.combiner.state_dict(),
        artifact_type="torch",
        description="Final model combiner state dictionary",
    )

    print("Training Complete! 🎉")

    return 0
