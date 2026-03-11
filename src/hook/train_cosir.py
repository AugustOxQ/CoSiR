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
    initialize_conditions,
    get_umap,
    visualize_ideal_condition_space,
    get_umap_recursive,
)
from src.metrics import CoSiRLoss


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

    # Initialize K conditions
    prototype_conditions = initialize_conditions(
        K=cfg.train.representative_number,
        method=cfg.train.initialization_method,
        scale=cfg.train.initialization_scale,
    )

    model = CoSiRModel(
        label_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.hidden_dim,
        num_conditions=cfg.train.representative_number,
        prototype_conditions=prototype_conditions,
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.model.clip_model, use_fast=False)

    # Setup criteria and optimizer and scheduler
    print("Initializing criteria and optimizer and scheduler")
    criteria = CoSiRLoss(
        temperature=cfg.loss.temperature,
        lambda_margin=cfg.loss.lambda_margin,
        lambda_infonce=cfg.loss.lambda_infonce,
        lambda_diversity=cfg.loss.lambda_diversity,
        lambda_preserve=cfg.loss.lambda_preserve,
        return_dict=cfg.loss.return_dict,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
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
                chunk_img, chunk_txt, img_full, txt_full = model.encode_img_txt(
                    image_inputs, text_inputs
                )  # DEBUG: txt_full is not used

                chunk_img, chunk_txt, img_full, txt_full = (
                    chunk_img.detach().cpu(),
                    chunk_txt.detach().cpu(),
                    img_full.detach().cpu(),
                    txt_full.detach().cpu(),
                )

                # print(chunk_img.shape, chunk_txt.shape, txt_full.shape)

                # Add features to feature manager
                feature_manager.add_features_chunk(
                    batch_id,
                    chunk_img,
                    chunk_txt,
                    img_full,
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

    # Set save path for embeddings
    evaluator.test_evaluator.processor.set_save_path(
        str(experiment.directory / "embeddings")
    )

    # clustering = Clustering(device=device)
    umap_vis = UMAP_vis(device=device)

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

    conditions_cache = None

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
            # txt_full = features_data["txt_full"].to(device, non_blocking=True)
            img_full = features_data["img_full"].to(device, non_blocking=True)

            comb_emb = model.combine(
                txt_features,
                img_features,
                img_full,
            )

            loss_dict = criteria(
                img_features,
                txt_features,
                comb_emb,
                model.combiner.conditions,
            )

            loss = loss_dict["total_loss"]

            epoch_loss += loss.item()
            num_batches += 1

            # Log batch-level loss components
            batch_log_dict = {
                f"train/batch_{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
            }

            with torch.no_grad():
                # 1. conditions的平均cosine similarity（应该接近0）
                normed = torch.nn.functional.normalize(
                    model.combiner.conditions, dim=-1
                )  # [K, dim]
                sim = normed @ normed.T  # [K, K]
                mask = ~torch.eye(model.combiner.conditions.shape[0], dtype=torch.bool)
                avg_conditions_sim = sim[mask].mean()  # 越小越好

                # 2. conditioned_txt和raw_txt的相似度
                cos_sim_txt_conditioned_txt = torch.nn.functional.cosine_similarity(
                    comb_emb, txt_features, dim=-1
                ).mean()

                # 3. 确认网络中gamma和beta的指
                gamma = model.combiner.gamma_scalar.get_newest()
                beta = model.combiner.beta_scalar.get_newest()
                entropy = model.combiner.entropy_scalar.get_newest()
                dynamic_condition_sim = (
                    model.combiner.dynamic_condition_scalar.get_newest()
                )

                if batch_idx % 200 == 0:

                    tmp_topk = 1
                    txt_features_norm = torch.nn.functional.normalize(
                        txt_features, dim=-1
                    )
                    img_features_norm = torch.nn.functional.normalize(
                        img_features, dim=-1
                    )
                    comb_emb_norm = torch.nn.functional.normalize(comb_emb, dim=-1)
                    sim_origin = txt_features_norm @ img_features_norm.T
                    sim_conditioned = comb_emb_norm @ img_features_norm.T

                    # Compute the difference between two similarity matrices's diagnoal mean
                    diff_mean = (
                        sim_conditioned.diagonal().mean() - sim_origin.diagonal().mean()
                    )

                    # Compute the differencebetween two similarity matrices's off-diagonal mean
                    diff_off_mean = (
                        sim_conditioned[
                            ~torch.eye(sim_conditioned.shape[0], dtype=torch.bool)
                        ].mean()
                        - sim_origin[
                            ~torch.eye(sim_origin.shape[0], dtype=torch.bool)
                        ].mean()
                    )

                    batch_log_dict.update(
                        {
                            "train_diagnostics/diagnoal_diff": diff_mean,
                            "train_diagnostics/off_diagnoal_diff_mean": diff_off_mean,
                        }
                    )

                    # rank_origin = torch.argsort(sim_origin, dim=-1, descending=True)
                    # rank_conditioned = torch.argsort(
                    #     sim_conditioned, dim=-1, descending=True
                    # )

                    # topk_origin = rank_origin[:, :tmp_topk]
                    # correct_origin = torch.zeros((1,), dtype=torch.bool, device=device)
                    # for i in range(tmp_topk):
                    #     contains_index = torch.eq(
                    #         topk_origin[i],
                    #         torch.arange(img_features.shape[0])
                    #         .unsqueeze(-1)
                    #         .to(device),
                    #     ).any()
                    #     correct_origin = torch.logical_or(
                    #         correct_origin, contains_index
                    #     )
                    # num_correct_origin = correct_origin.sum().item()
                    # recall_origin = num_correct_origin

                    # topk_conditioned = rank_conditioned[:, :tmp_topk]
                    # correct_conditioned = torch.zeros(
                    #     (1,), dtype=torch.bool, device=device
                    # )
                    # for i in range(tmp_topk):
                    #     contains_index = torch.eq(
                    #         topk_conditioned[i],
                    #         torch.arange(img_features.shape[0])
                    #         .unsqueeze(-1)
                    #         .to(device),
                    #     ).any()
                    #     correct_conditioned = torch.logical_or(
                    #         correct_conditioned, contains_index
                    #     )
                    # num_correct_conditioned = correct_conditioned.sum().item()
                    # recall_conditioned = num_correct_conditioned

                    # recall_diff = recall_conditioned - recall_origin

                batch_log_dict.update(
                    {
                        "train_diagnostics/avg_conditions_sim": avg_conditions_sim,
                        "train_diagnostics/cos_sim_txt_conditioned_txt": cos_sim_txt_conditioned_txt,
                        "train_diagnostics/gamma": gamma,
                        "train_diagnostics/beta": beta,
                        "train_diagnostics/entropy": entropy,
                        "train_diagnostics/dynamic_condition_sim": dynamic_condition_sim,
                    }
                )

            batch_log_dict.update(
                {
                    "train_extra/epoch": epoch,
                    "train_extra/batch": batch_idx,
                    "train_extra/step": global_step,
                }
            )
            logger.log_metrics(batch_log_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model parameters with accumulated state
            global_step += 1

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
                "train_extra/epoch": epoch,
                "train_extra/loss": avg_loss,
                "train_extra/epoch_time": epoch_time,
            }
        )

        scheduler.step()

        # Check whether label embeddings are 2d or higher dimensional
        prototype_conditions_copy = model.combiner.conditions.clone().detach().cpu()
        # Add to cache
        if conditions_cache is None:
            conditions_cache = prototype_conditions_copy
        else:
            conditions_cache = torch.cat(
                [conditions_cache, prototype_conditions_copy], dim=0
            )

        if cfg.eval.perform_evaluation and (
            cfg.train.epochs == 0
            or epoch % evaluation_config.evaluation_interval == 0
            or epoch == cfg.train.epochs - 1
        ):
            model.eval()
            with torch.no_grad():
                # Test evaluation
                test_results_detailed = evaluator.evaluate_test(
                    model=model,
                    processor=processor,
                    dataloader=test_loader,
                    epoch=epoch,
                    return_detailed_results=True,  # 记住这里是true，也就是detailed_results
                    use_oracle=False,
                )

                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    test_results,
                ) = test_results_detailed  # type: ignore

                all_raw_image = test_set.get_all_raw_image()

                # Log test results
                for metric, value in test_results.metrics.items():
                    logger.log_metrics({metric: value})

                # Visualize test results
                if conditions_cache.shape[1] == 2:
                    umap_features = conditions_cache.cpu().numpy()
                else:
                    umap_features = umap_vis.learn_umap(
                        conditions_cache, close_cluster=True
                    )

                # Visualize label embeddings
                fig = get_umap_recursive(
                    umap_features,
                    umap_labels=None,
                    epoch=epoch,
                    no_outlier=True,
                    samples_to_track=[0, 1, 2, 3, 4],
                    num_repeats=cfg.train.representative_number,
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

                # cosir_automatic_evaluator = CoSiRAutomaticEvaluator(
                #     model,
                #     (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
                #     label_embeddings_all,
                #     device,
                # )
                # result = cosir_automatic_evaluator.evaluate_all()
                # logger.log_metrics(result)

    experiment.save_artifact(
        name="phase_1_model",
        folder="checkpoints",
        data=model.combiner.state_dict(),
        artifact_type="torch",
        description="Phase 1 model combiner state dictionary",
    )

    print("Training Complete!")

    return 0
