import time
import os
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoProcessor
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
)
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
    visualize_given_conditions_image_to_text,
    visualize_given_conditions_text_to_image,
    CoSiRAutomaticEvaluator,
)
from src.metrics import LabelContrastiveLoss_enhance


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
        backbone_model=cfg.model.clip_model,
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

    if cfg.scheduler.type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.T_max if cfg.scheduler.T_max > 0 else cfg.train.epochs,
            eta_min=cfg.scheduler.eta_min,
            last_epoch=-1,
        )
    elif cfg.scheduler.type == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.T_0,
            T_mult=cfg.scheduler.T_mult if cfg.scheduler.T_mult > 0 else 1,
            eta_min=cfg.scheduler.eta_min,
            last_epoch=-1,
        )
    elif cfg.scheduler.type == "LinearLR":
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=int(cfg.train.epochs * 0.1),
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1e-5,
            total_iters=int(cfg.train.epochs * 0.9),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[int(cfg.train.epochs * 0.1)],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")
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
        feature_manager.validate_backbone(cfg.model.clip_model)
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

        feature_manager.open_for_writing(
            len(pre_extraction_dataset),
            feature_dims,
            backbone_model=cfg.model.clip_model,
        )

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

                # Release fragmented reserved-but-unallocated GPU memory each batch.
                # Especially important for large-patch models (SigLIP, CLIP-L/14).
                torch.cuda.empty_cache()

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

    # Warm-up auto-end tracking
    automatic_warm_up_end = False
    warm_up_loss_history: list[float] = []
    in_warmup = (
        True if cfg.train.warm_up_epochs > 0 else False
    )  # single flag; transitions to False exactly once

    for epoch in range(cfg.train.epochs):
        experiment.current_epoch = epoch

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Transition out of warm-up: hard limit (if > 0) OR automatic plateau trigger
        # warm_up_epochs=0 means no hard limit — rely solely on automatic detection
        scheduled_end = (
            cfg.train.warm_up_epochs > 0 and epoch >= cfg.train.warm_up_epochs
        )
        if in_warmup and (scheduled_end or automatic_warm_up_end):
            in_warmup = False
            reason = "scheduled" if scheduled_end else "auto (plateau)"
            print(
                f"[WarmUp] Warm-up ended at epoch {epoch} ({reason}) — freezing combiner, starting embedding updates"
            )
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

            label_embeddings_clone = label_embeddings_data.clone().detach()
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

            if batch_idx % 100 == 0:
                # Compare comb_emb and txt_features
                cos_sim = torch.nn.functional.cosine_similarity(
                    comb_emb,
                    torch.nn.functional.normalize(txt_features, dim=-1),
                    dim=-1,
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
                    "train/cos_sim": (
                        cos_sim.mean().item() if cos_sim is not None else None
                    ),
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
            if not in_warmup:
                if cfg.train.normalize:
                    label_embeddings = torch.nn.functional.normalize(
                        label_embeddings, dim=-1
                    )

                embedding_manager.update_embeddings(batch_sample_ids, label_embeddings)

                label_embeddings_diff = (
                    (label_embeddings.cpu() - label_embeddings_clone.cpu())
                    .norm(dim=-1)
                    .mean()
                )
                batch_log_dict.update(
                    {
                        "train/label_embeddings_diff": label_embeddings_diff.item(),
                    }
                )
                logger.log_metrics(batch_log_dict)

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
                "train/in_warmup": int(in_warmup),
            }
        )

        # Auto warm-up end: track plateau during warm-up phase
        if in_warmup and not automatic_warm_up_end:
            warm_up_loss_history.append(avg_loss)
            patience = cfg.train.warm_up_patience
            min_delta_pct = cfg.train.warm_up_min_delta_pct
            if len(warm_up_loss_history) > patience:
                ref_loss = warm_up_loss_history[-(patience + 1)]
                improvement_pct = (ref_loss - avg_loss) / (abs(ref_loss) + 1e-8) * 100
                if improvement_pct < min_delta_pct:
                    automatic_warm_up_end = True
                    print(
                        f"[WarmUp] Plateau detected: loss improved only {improvement_pct:.3f}% "
                        f"over last {patience} epochs (threshold: {min_delta_pct}%). "
                        f"Auto-ending warm-up after epoch {epoch}."
                    )
                    logger.log_metrics({"train/warmup_auto_ended_epoch": epoch})

        scheduler.step()

        if cfg.eval.perform_evaluation and (
            cfg.train.epochs == 0  # This is for test only
            or epoch % cfg.eval.evaluation_interval == 0
            or epoch == cfg.train.epochs - 1
        ):
            model.eval()
            with torch.no_grad():
                # Test evaluation
                torch.cuda.empty_cache()  # release any GPU residuals from training
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

                print("Visualizing ideal condition space")
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

                # print("Visualizing image-to-text")
                # figs_i2t = visualize_given_conditions_image_to_text(
                #     representatives.cpu().numpy(),
                #     model,
                #     (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                #     n_queries=1,
                #     k=3,
                #     all_raw_image=all_raw_image,
                #     device=device,
                # )
                # for q_idx, fig in enumerate(figs_i2t):
                #     experiment.save_artifact(
                #         name=f"cond_i2t_{epoch}_query{q_idx}",
                #         data=fig,
                #         artifact_type="figure",
                #         folder="plots",
                #         description=f"Image-to-text by condition, epoch {epoch}, query {q_idx}",
                #     )

                # print("Visualizing text-to-image")
                # figs_t2i = visualize_given_conditions_text_to_image(
                #     representatives.cpu().numpy(),
                #     model,
                #     (all_img_emb, all_txt_emb, all_raw_image, all_raw_text),
                #     n_queries=1,
                #     texts_per_image=test_set.captions_per_image,
                #     k=3,
                #     device=device,
                # )
                # for q_idx, fig in enumerate(figs_t2i):
                #     experiment.save_artifact(
                #         name=f"cond_t2i_{epoch}_query{q_idx}",
                #         data=fig,
                #         artifact_type="figure",
                #         folder="plots",
                #         description=f"Text-to-image by condition, epoch {epoch}, query {q_idx}",
                #     )

                # plt.close("all")

                print("Evaluating automatic evaluator")
                cosir_automatic_evaluator = CoSiRAutomaticEvaluator(
                    model,
                    (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                    label_embeddings_all,
                    device,
                    representatives=representatives,
                    hdbscan_labels=hdbscanlabels,
                )
                result = cosir_automatic_evaluator.evaluate_all()
                del cosir_automatic_evaluator  # free GPU tensors (conditions, image/text embs)
                torch.cuda.empty_cache()

                # Log only key scalar metrics to wandb; full results are printed by evaluate_all
                wandb_metrics = {}
                if "magnitude_effect" in result:
                    wandb_metrics["eval/magnitude_r"] = result["magnitude_effect"][
                        "correlation"
                    ]
                if "condition_distance_correlation" in result:
                    wandb_metrics["eval/condition_dist_rho"] = result[
                        "condition_distance_correlation"
                    ]["spearman_rho"]
                if "retrieval_gain" in result:
                    wandb_metrics["eval/R@1_gain"] = result["retrieval_gain"][
                        "R@1_absolute_gain"
                    ]
                    wandb_metrics["eval/R@1_baseline"] = result["retrieval_gain"][
                        "R@1_baseline"
                    ]
                    wandb_metrics["eval/R@1_conditional"] = result["retrieval_gain"][
                        "R@1_conditional"
                    ]
                if "diversity" in result:
                    wandb_metrics["eval/diversity_jsd"] = result["diversity"][
                        "mean_jsd"
                    ]
                if (
                    "best_condition_upper_bound" in result
                    and result["best_condition_upper_bound"]
                ):
                    wandb_metrics["eval/R@1_boost"] = result[
                        "best_condition_upper_bound"
                    ]["R@1_boost"]
                    wandb_metrics["eval/R@1_best_condition"] = result[
                        "best_condition_upper_bound"
                    ]["R@1_best_condition"]
                if "space_quality" in result:
                    wandb_metrics["eval/silhouette"] = result["space_quality"][
                        "silhouette_score"
                    ]
                    wandb_metrics["eval/n_effective_dims"] = result["space_quality"][
                        "n_effective_dims"
                    ]
                logger.log_metrics(wandb_metrics)

                # ─── Retrieval snapshot (qualitative cross-epoch tracking) ───
                print("Saving retrieval snapshot...")
                _N_FIXED = 50  # first N image/text queries, fixed across all epochs
                _TOP_K = 10  # top-K results to store per query

                snap_dir = experiment.directory / "retrieval_snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)

                _n_img = all_img_emb.shape[0]
                _n_txt = all_txt_emb.shape[0]
                _nfi = min(_N_FIXED, _n_img)
                _nft = min(_N_FIXED, _n_txt)

                _img_n = F.normalize(all_img_emb.to(device), dim=-1)  # [N_img, D]
                # Use raw (unnormalized) text embeddings — consistent with training
                _txt_raw = all_txt_emb.to(device)  # [N_txt, D]
                _fixed_img_n = _img_n[:_nfi]  # [_nfi, D]
                _fixed_txt_raw = _txt_raw[:_nft]  # [_nft, D]

                _run_max_i2t = None  # [_nfi, N_txt]
                _run_max_t2i = None  # [_nft, N_img]

                for _rep in representatives:
                    _cond = _rep.unsqueeze(0).to(device)
                    # Modulate all texts for i2t query similarity
                    # combine() already outputs unit-normalized embeddings
                    _txt_mod = model.combine(_txt_raw, None, _cond.expand(_n_txt, -1))
                    _sim_i2t = (_fixed_img_n @ _txt_mod.T).cpu()
                    _run_max_i2t = (
                        _sim_i2t
                        if _run_max_i2t is None
                        else torch.maximum(_run_max_i2t, _sim_i2t)
                    )

                    # Modulate only fixed text queries for t2i efficiency
                    _txt_mod_fixed = model.combine(
                        _fixed_txt_raw, None, _cond.expand(_nft, -1)
                    )
                    _sim_t2i = (_txt_mod_fixed @ _img_n.T).cpu()
                    _run_max_t2i = (
                        _sim_t2i
                        if _run_max_t2i is None
                        else torch.maximum(_run_max_t2i, _sim_t2i)
                    )

                _ki2t = min(_TOP_K, _n_txt)
                _kt2i = min(_TOP_K, _n_img)
                _top_i2t = torch.topk(_run_max_i2t, k=_ki2t, dim=1).indices  # [_nfi, K]
                _top_t2i = torch.topk(_run_max_t2i, k=_kt2i, dim=1).indices  # [_nft, K]

                # Ground-truth masks
                _ittmap = image_to_text_map.cpu()  # [N_img, cpi]
                _ttimap = text_to_image_map.cpu()  # [N_txt]

                _is_gt_i2t = torch.zeros(_nfi, _ki2t, dtype=torch.bool)
                for _q in range(_nfi):
                    _gt_set = set(_ittmap[_q].tolist())
                    for _kp, _tidx in enumerate(_top_i2t[_q].tolist()):
                        if _tidx in _gt_set:
                            _is_gt_i2t[_q, _kp] = True

                _is_gt_t2i = torch.zeros(_nft, _kt2i, dtype=torch.bool)
                for _q in range(_nft):
                    _gt_img = _ttimap[_q].item()
                    for _kp, _iidx in enumerate(_top_t2i[_q].tolist()):
                        if _iidx == _gt_img:
                            _is_gt_t2i[_q, _kp] = True

                torch.save(
                    {
                        "epoch": epoch,
                        "i2t": {
                            "query_indices": list(range(_nfi)),
                            "top_k": _top_i2t,
                            "is_gt": _is_gt_i2t,
                        },
                        "t2i": {
                            "query_indices": list(range(_nft)),
                            "top_k": _top_t2i,
                            "is_gt": _is_gt_t2i,
                        },
                    },
                    snap_dir / f"epoch_{epoch:04d}.pt",
                )
                print(
                    f"  Saved retrieval snapshot → {snap_dir / f'epoch_{epoch:04d}.pt'}"
                )

                # Metadata: save once (captions, GT maps, image paths, CLIP baseline)
                _meta_path = snap_dir / "metadata.pt"
                if not _meta_path.exists():
                    _image_paths = [
                        os.path.join(
                            test_set.image_path, test_set.annotations[i]["image"]
                        )
                        for i in range(_n_img)
                    ]
                    # CLIP baseline (no condition): raw cosine similarity
                    _txt_n_for_clip = F.normalize(_txt_raw, dim=-1)
                    _clip_sim_i2t = (
                        _fixed_img_n @ _txt_n_for_clip.T
                    ).cpu()  # [_nfi, N_txt]
                    _clip_sim_t2i = (
                        F.normalize(_fixed_txt_raw, dim=-1) @ _img_n.T
                    ).cpu()  # [_nft, N_img]
                    _clip_top_i2t = torch.topk(_clip_sim_i2t, k=_ki2t, dim=1).indices
                    _clip_top_t2i = torch.topk(_clip_sim_t2i, k=_kt2i, dim=1).indices

                    _clip_is_gt_i2t = torch.zeros(_nfi, _ki2t, dtype=torch.bool)
                    for _q in range(_nfi):
                        _gt_set = set(_ittmap[_q].tolist())
                        for _kp, _tidx in enumerate(_clip_top_i2t[_q].tolist()):
                            if _tidx in _gt_set:
                                _clip_is_gt_i2t[_q, _kp] = True

                    _clip_is_gt_t2i = torch.zeros(_nft, _kt2i, dtype=torch.bool)
                    for _q in range(_nft):
                        _gt_img = _ttimap[_q].item()
                        for _kp, _iidx in enumerate(_clip_top_t2i[_q].tolist()):
                            if _iidx == _gt_img:
                                _clip_is_gt_t2i[_q, _kp] = True

                    torch.save(
                        {
                            "captions": all_raw_text,
                            "image_to_text_map": _ittmap,
                            "text_to_image_map": _ttimap,
                            "captions_per_image": test_set.captions_per_image,
                            "n_images": _n_img,
                            "n_texts": _n_txt,
                            "image_paths": _image_paths,
                            "clip_baseline": {
                                "i2t": {
                                    "top_k": _clip_top_i2t,
                                    "is_gt": _clip_is_gt_i2t,
                                },
                                "t2i": {
                                    "top_k": _clip_top_t2i,
                                    "is_gt": _clip_is_gt_t2i,
                                },
                            },
                        },
                        _meta_path,
                    )
                    print(f"  Saved retrieval metadata → {_meta_path}")

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
