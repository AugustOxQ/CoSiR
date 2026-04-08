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
from itertools import chain

from src.dataset import (
    CoSiRShardDataset,
    CoSiRShardStreamDataset,
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
    get_representatives_hdbscan,
    get_umap,
    visualize_ideal_condition_space,
    visualize_angular_semantics_fast,
    visualize_angular_semantics_text_to_image_fast,
    CoSiRAutomaticEvaluator,
)
from src.metrics import LabelContrastiveLoss_enhance, LabelPredictionLoss


def train_cosir(cfg, logger):
    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    storage_dir = cfg.featuremanager.storage_dir

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
        num_conditions=cfg.train.representative_number,
        dropout=cfg.model.dropout,
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
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "condition_predictor" not in n
                ],
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.T_max if cfg.scheduler.T_max > 0 else cfg.train.epochs,
        eta_min=cfg.scheduler.eta_min,
        last_epoch=-1,
    )
    # Check either load existing sample ids
    metadata_path = os.path.join(storage_dir, "metadata.json")
    if os.path.exists(metadata_path) and cfg.train.load_existing_features:
        print("Loading existing feature store")
        feature_manager = FeatureManager(
            storage_dir,
            shard_size=cfg.featuremanager.shard_size,
            hdf5_compression=cfg.featuremanager.hdf5_compression,
            hdf5_compression_level=cfg.featuremanager.hdf5_compression_level,
        )
        sample_ids_list = feature_manager.get_all_sample_ids()
        print(f"Loaded {len(sample_ids_list):,} sample ids from existing store")
    else:
        print("Extracting features")
        feature_manager = FeatureManager(
            storage_dir,
            shard_size=cfg.featuremanager.shard_size,
            hdf5_compression=cfg.featuremanager.hdf5_compression,
            hdf5_compression_level=cfg.featuremanager.hdf5_compression_level,
        )

        # Build extraction dataset
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

        # Probe first batch to determine feature dimensions
        print("Probing feature dimensions…")
        _probe_loader = DataLoader(
            pre_extraction_dataset, batch_size=2, shuffle=False, num_workers=0
        )
        _img_in, _txt_in, _ = next(iter(_probe_loader))
        _img_in = _img_in.to(device)
        _txt_in = {k: v.to(device) for k, v in _txt_in.items()}
        with torch.no_grad():
            _img_e, _txt_e, _img_f, _txt_f = model.encode_img_txt(_img_in, _txt_in)
        feature_dims = {
            "img_features": tuple(_img_e.shape[1:]),
            "txt_features": tuple(_txt_e.shape[1:]),
        }
        if cfg.featuremanager.store_img_full:
            feature_dims["img_full"] = tuple(_img_f.shape[1:])
        if cfg.featuremanager.store_txt_full:
            feature_dims["txt_full"] = tuple(_txt_f.shape[1:])
        print(f"Feature dims: {feature_dims}")
        del _img_in, _txt_in, _img_e, _txt_e, _img_f, _txt_f, _probe_loader

        feature_manager.open_for_writing(len(pre_extraction_dataset), feature_dims)

        pre_extraction_dataloader = DataLoader(
            pre_extraction_dataset,
            batch_size=cfg.featuremanager.extraction_batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )

        with torch.no_grad():
            for batch in tqdm(pre_extraction_dataloader, desc="Extracting features"):
                image_inputs, text_inputs, sample_ids = batch
                sample_ids = [int(s) for s in sample_ids]
                image_inputs = image_inputs.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                img_e, txt_e, img_f, txt_f = model.encode_img_txt(
                    image_inputs, text_inputs
                )

                feature_manager.write_batch(
                    img_e,
                    txt_e,
                    sample_ids,
                    img_full=img_f if cfg.featuremanager.store_img_full else None,
                    txt_full=txt_f if cfg.featuremanager.store_txt_full else None,
                )

        feature_manager.finalize_writing()
        sample_ids_list = feature_manager.get_all_sample_ids()

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

    clustering = Clustering(device=device)
    umap_vis = UMAP_vis(device=device)

    # ========== OPTIMIZED: TrainableEmbeddingManager with Intelligent Caching ==========

    embedding_manager = TrainableEmbeddingManager(
        sample_ids=sample_ids_list,
        embedding_dim=cfg.model.embedding_dim,
        storage_mode=cfg.embeddingmanager.storage_mode,
        device=device,
        initialization_strategy=cfg.embeddingmanager.initialization_strategy,
        embeddings_dir=str(experiment.directory / "training_embeddings"),
        cache_l1_size_mb=cfg.embeddingmanager.cache_l1_size_mb,
        cache_l2_size_mb=cfg.embeddingmanager.cache_l2_size_mb,
        enable_l3_cache=cfg.embeddingmanager.enable_l3_cache,
        auto_sync=cfg.embeddingmanager.auto_sync,
        sync_batch_size=cfg.embeddingmanager.sync_batch_size,
        chunk_size=cfg.embeddingmanager.embedding_chunk_size,
    )

    # ========== Template Embedding Initialization ==========
    if (
        cfg.train.initialization_strategy == "imgtxt"
        or cfg.train.initialization_strategy == "txt"
        or cfg.train.initialization_strategy == "img"
    ):
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
            # Initialize embeddings with different strategy
            if cfg.train.initialization_strategy == "imgtxt":
                print("Initializing embeddings with imgtxt strategy...")
                embedding_manager.initialize_embeddings_imgtxt(
                    feature_manager,
                    model,
                    device,
                    factor=cfg.train.imgtxt_factor,
                    normalize=cfg.train.normalize,
                )
            elif cfg.train.initialization_strategy == "txt":
                print("Initializing embeddings with txt strategy...")
                embedding_manager.initialize_embeddings_txt(
                    feature_manager,
                    model,
                    device,
                    factor=cfg.train.imgtxt_factor,
                    normalize=cfg.train.normalize,
                )
            elif cfg.train.initialization_strategy == "img":
                print("Initializing embeddings with img strategy...")
                embedding_manager.initialize_embeddings_img(
                    feature_manager,
                    model,
                    device,
                    factor=cfg.train.imgtxt_factor,
                    normalize=cfg.train.normalize,
                )
            else:
                raise ValueError(
                    f"Unknown initialization strategy: {cfg.train.initialization_strategy}"
                )

            # Save as template embeddings if enabled (default: True)
            if getattr(cfg.train, "save_as_template_embeddings", True):
                print("Storing embeddings as template for future use...")
                embedding_manager.store_imgtxt_template()

    # ── Training dataset: auto-select RAM vs streaming based on available memory ──
    feature_types = (
        feature_manager.available_features
    )  # e.g. ['img_features','txt_features']
    if feature_manager.fits_in_ram():
        print(
            f"RAM mode: loading {feature_manager.cls_features_size_gb():.1f} GiB of "
            "features into RAM for true-random batches."
        )
        train_set = CoSiRShardDataset(feature_manager, feature_types=feature_types)
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
        )
    else:
        print(
            f"Stream mode: {feature_manager.cls_features_size_gb():.1f} GiB does not "
            "fit in RAM — using shard-streaming with shuffle window."
        )
        train_set = CoSiRShardStreamDataset(
            feature_manager,
            feature_types=feature_types,
            window_shards=cfg.featuremanager.shuffle_window_shards,
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
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

    sample_types = []
    # Load sample types:
    if cfg.data.dataset_type == "impressions":
        print("Loading sample types for Impressions dataset")
        import json

        train_file_path = cfg.data.train_annotation_path
        train_file = json.load(open(train_file_path))

        # reorder the train_file based on the sample_ids_list
        train_file = [train_file[i] for i in sample_ids_list]

        # Collect sample types
        # sample_types = []
        for item in train_file:
            type_str = item["caption_type"]

            if "caption" in type_str:
                type_int = 0
            elif "description" in type_str:
                type_int = 1
            elif "impression" in type_str:
                type_int = 2
            elif "aesthetic" in type_str:
                type_int = 3
            else:
                raise ValueError(f"Unknown caption type: {type_str}")

            sample_types.append(type_int)

        sample_types = np.array(sample_types)

    global_step = 0

    for epoch in range(cfg.train.epochs):
        experiment.current_epoch = epoch

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Freeze model parameters and only train label embeddings
        if epoch == cfg.train.warm_up_epochs:
            for param in model.combiner.parameters():
                param.requires_grad = False

        # Track epoch start time for performance monitoring
        epoch_start_time = time.time()

        if isinstance(train_set, CoSiRShardStreamDataset):
            train_set.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            img_features = batch["img_features"].to(device, non_blocking=True)
            txt_features = batch["txt_features"].to(device, non_blocking=True)
            # txt_full is optional — pass zeros if not stored
            if "txt_full" in batch:
                txt_full = batch["txt_full"].to(device, non_blocking=True)
            else:
                txt_full = torch.zeros_like(txt_features)
            batch_sample_ids = batch["sample_ids"].tolist()

            # Load label embeddings for this batch's sample IDs
            label_embeddings_data = embedding_manager.get_embeddings(batch_sample_ids)
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

            loss_dict = criteria(
                img_features,
                txt_features,
                comb_emb,
                None,
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

            # Persist updated label embeddings back to the manager

            if cfg.train.normalize:
                label_embeddings = torch.nn.functional.normalize(
                    label_embeddings, dim=-1
                )

            embedding_manager.update_embeddings(batch_sample_ids, label_embeddings)

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

        if cfg.eval.perform_evaluation and (
            cfg.train.epochs == 0  # This is for test only
            or epoch % cfg.eval.evaluation_interval == 0
            or epoch == cfg.train.epochs - 1
        ):
            model.eval()
            with torch.no_grad():
                # Test evaluation
                print("Getting all embeddings")
                _, label_embeddings_all = embedding_manager.get_all_embeddings()
                print("Getting HDBSCAN labels")
                hdbscanlabels, _ = clustering.get_hdbscan(
                    label_embeddings_all.cpu(), method="eom"
                )
                print("Getting representatives")
                representatives = get_representatives_hdbscan(
                    hdbscanlabels,
                    label_embeddings_all.cpu(),
                    (
                        cfg.train.representative_number
                        if epoch != cfg.train.epochs - 1
                        else 30  # Use higher number of representatives for the last epoch
                    ),
                )
                print(f"Evaluating with {len(representatives)} representatives")
                test_results_detailed = evaluator.evaluate_test(
                    model=model,
                    processor=processor,
                    dataloader=test_loader,
                    label_embeddings=representatives,  # Your label embeddings
                    epoch=epoch,
                    return_detailed_results=True,  # 记住这里是true，也就是detailed_results
                    use_oracle=True,
                    oracle_aggregation=cfg.eval.oracle_aggregation,
                )

                (
                    all_img_emb,
                    all_txt_emb,
                    all_raw_text,
                    text_to_image_map,
                    image_to_text_map,
                    test_results,
                ) = test_results_detailed  # type: ignore

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
                    umap_labels=hdbscanlabels,
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

                if len(sample_types) == len(umap_features):
                    print("Get ground truth sample types")
                    fig_3 = get_umap(
                        umap_features,
                        umap_labels=sample_types,
                        epoch=epoch,
                        no_outlier=True,
                        samples_to_track=[0, 1, 2, 3, 4],
                    )

                    experiment.save_artifact(
                        name=f"ground_truth_sample_types_{epoch}",
                        data=fig_3,
                        artifact_type="figure",
                        folder="plots",
                        description=f"Ground truth sample types visualization at epoch {epoch}",
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
                        "vis/ground_truth_sample_types": (
                            wandb.Image(fig_3)
                            if len(sample_types) == len(umap_features)
                            else None
                        ),
                    }
                )

                plt.close("all")

                # all_raw_image = test_set.get_all_raw_image()

                # for tmp_round in range(3):
                #     fig3 = visualize_angular_semantics_fast(
                #         label_embeddings_all.cpu().numpy(),
                #         model,
                #         (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                #         device=device,
                #     )
                #     experiment.save_artifact(
                #         name=f"angular_semantics_fast_{epoch}_tmp_{tmp_round}",
                #         data=fig3,
                #         artifact_type="figure",
                #         folder="plots",
                #         description=f"Angular semantics visualization at epoch {epoch}",
                #     )

                #     fig4 = visualize_angular_semantics_text_to_image_fast(
                #         label_embeddings_all.cpu().numpy(),
                #         model,
                #         (
                #             all_img_emb,
                #             all_txt_emb,
                #             all_raw_image,
                #             all_raw_text,
                #         ),
                #         device=device,
                #     )

                #     experiment.save_artifact(
                #         name=f"angular_semantics_text_to_image_{epoch}_tmp_{tmp_round}",
                #         data=fig4,
                #         artifact_type="figure",
                #         folder="plots",
                #         description=f"Angular semantics visualization (text-to-image) at epoch {epoch}",
                #     )

                #     plt.close("all")

                cosir_automatic_evaluator = CoSiRAutomaticEvaluator(
                    model,
                    (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                    label_embeddings_all,
                    device,
                )
                result = cosir_automatic_evaluator.evaluate_all()
                logger.log_metrics(result)

    # ========== TRAINING COMPLETE: Final Performance Summary ==========
    # Copy final embeddings (memmap files) to a snapshot directory for phase-2 use
    import pathlib

    embedding_manager._copy_to(pathlib.Path(experiment.directory) / "final_embeddings")

    # Save sample_ids list so phase-2 scripts can reconstruct the id→position mapping
    experiment.save_artifact(
        name="sample_ids",
        data=embedding_manager.sample_ids,
        artifact_type="pickle",
        description="Ordered sample ID list (position-indexed, replaces chunk_mapping/id_to_chunk_index)",
        folder="embeddings",
    )

    experiment.save_artifact(
        name="phase_1_model",
        folder="checkpoints",
        data=model.combiner.state_dict(),
        artifact_type="torch",
        description="Phase 1 model combiner state dictionary",
    )

    print("Training Complete!")

    return 0
