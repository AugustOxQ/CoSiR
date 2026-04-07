"""
This file is used to take checkpoint from phase 1 and train the condition predictor only for phase 2.
"""

import time
from pathlib import Path
import os
import torch
import random
import pickle
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
    TrainableEmbeddingManager,
    precompute_nearest_condition_labels,
    get_representatives_polar_grid,
    get_umap,
    visualize_ideal_condition_space,
    CoSiRAutomaticEvaluator,
    get_representatives_polar_grid_outsideonly,
)
from src.metrics import (
    LabelContrastiveLoss_enhance,
    LabelPredictionLoss,
    LabelClassificationLoss,
)


def train_cosir_phase2(cfg, logger):
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
            "chunk_size": cfg.featuremanager.shard_size,
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
        num_conditions=cfg.train.representative_number,
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.model.clip_model, use_fast=False)

    # Load phase 1 model
    phase_1_model_path = (
        "/project/CoSiR/res/CoSiR_Experiment/coco/checkpoints/phase_1_model.pt"
    )

    # Load everything except the condition predictors, because the condition predictors are already initialized in phase 1.
    phase_1_model = torch.load(phase_1_model_path, map_location=device)
    model.combiner.load_state_dict(phase_1_model, strict=False)

    # Setup criteria and optimizer and scheduler
    print("Initializing criteria and optimizer and scheduler")

    criteria_2 = LabelClassificationLoss(
        total_epochs=cfg.train.epochs_2,
        warm_up_epochs=cfg.loss.warm_up_epochs,
        middle_epochs=cfg.loss.middle_epochs,
        return_dict=cfg.loss.return_dict,
    )

    optimizer_2 = torch.optim.AdamW(
        [
            {
                "params": chain(
                    model.unified_condition_predictor.parameters(),
                ),
                "lr": cfg.optimizer.lr_2,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ]
    )

    scheduler_2 = CosineAnnealingLR(
        optimizer_2,
        T_max=cfg.scheduler.T_max if cfg.scheduler.T_max > 0 else cfg.train.epochs_2,
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

    # Initialize folder manager
    exp_manager = ExperimentManager(cfg.experiment.results_dir)

    # Create comprehensive experiment
    experiment = exp_manager.create_experiment(
        name=cfg.experiment.name,
        config=cfg,
        tags=cfg.experiment.tags,
        description=cfg.experiment.description,
    )

    # Initialize evaluator
    evaluator = EvaluationManager(evaluation_config)

    print(f"Created experiment: {experiment.name}")
    print(f"Experiment directory: {experiment.directory}")

    # ========== OPTIMIZED: TrainableEmbeddingManager with Intelligent Caching ==========

    # Auto-optimize chunk size based on batch size if not specified
    embedding_chunk_size = cfg.embeddingmanager.embedding_chunk_size
    if embedding_chunk_size is None:
        # Optimal chunk size: 4x batch size, capped at reasonable limits
        embedding_chunk_size = cfg.featuremanager.shard_size
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

    # Load existing embeddings if exists
    phase_1_embeddings_path = (
        "/project/CoSiR/res/CoSiR_Experiment/coco/final_embeddings"
    )

    phase_1_other_path = "/project/CoSiR/res/CoSiR_Experiment/coco/embeddings"

    embedding_manager.load_phase_1_template(phase_1_embeddings_path)

    # (chunk_mapping / id_to_chunk_index pkl files no longer used — embeddings are
    #  stored as flat memmap and the id→position index is rebuilt from sample_ids.npy)

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
        batch_size=cfg.featuremanager.shard_size,  # Load one shard at a time (batch_idx only)
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

    global_step = cfg.train.epochs

    print("Extracting pre-trained condition embeddings")
    # First extract all pre-trained condition embeddings
    _, label_embeddings_all = embedding_manager.get_all_embeddings()

    print("Getting pre-trained representatives")
    pre_trained_representatives = get_representatives_polar_grid_outsideonly(
        label_embeddings_all.cpu(),
        cfg.train.representative_number,
    )  # This will be used to train a classifier to predict the condition, now in total 12x3 = 36 representatives

    pre_trained_representatives_device = pre_trained_representatives.to(device)

    model.set_pretrained_representatives(pre_trained_representatives, device)

    # Start label prediction loss training
    print("Starting label prediction loss training")
    for epoch in range(cfg.train.epochs_2):
        model.train()
        # Freeze the combiner
        for param in model.combiner.parameters():
            param.requires_grad = False

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

            # Get embeddings for this shard by sample_id
            batch_sample_ids = features_data["sample_ids"].tolist()
            label_embeddings = embedding_manager.get_embeddings(batch_sample_ids).to(device)

            batch_labels = precompute_nearest_condition_labels(
                label_embeddings, pre_trained_representatives_device
            )

            # # Start both prediction
            # pred_cond_both, logits_both = model.predict_condition(
            #     img_features,
            #     txt_features,
            #     "imgtxt",
            #     training_phase=True,
            # )
            # comb_emb_both = model.combine(
            #     txt_features,
            #     None,
            #     pred_cond_both,
            #     epoch=epoch,
            #     return_label_proj=False,
            # )
            # loss_dict_both = criteria_2(
            #     logits_both,
            #     pred_cond_both,
            #     img_features,
            #     txt_features,
            #     batch_labels,
            #     comb_emb_both,
            #     epoch,
            # )
            # loss_both = loss_dict_both["total_loss"]
            # batch_log_dict_both = {
            #     f"train2_both/{k}": v.item() if torch.is_tensor(v) else v
            #     for k, v in loss_dict_both.items()
            # }
            # logger.log_metrics(batch_log_dict_both)

            # See text only
            pred_cond_img, logits_img = model.predict_condition(
                None,
                txt_features,
                "img",
                training_phase=True,
            )

            comb_emb_img = model.combine(
                txt_features,
                None,
                pred_cond_img,
                epoch=epoch,
                return_label_proj=False,
            )
            loss_dict_img = criteria_2(
                logits_img,
                pred_cond_img,
                img_features,
                txt_features,
                batch_labels,
                comb_emb_img,
                epoch,
            )
            loss_img = loss_dict_img["total_loss"]
            batch_log_dict_img = {
                f"train2_txt/{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict_img.items()
            }
            logger.log_metrics(batch_log_dict_img)

            # See image only
            pred_cond_txt, logits_txt = model.predict_condition(
                img_features,
                None,
                "txt",
                training_phase=True,
            )
            comb_emb_txt = model.combine(
                txt_features,
                None,
                pred_cond_txt,
                epoch=epoch,
                return_label_proj=False,
            )
            loss_dict_txt = criteria_2(
                logits_txt,
                pred_cond_txt,
                img_features,
                txt_features,
                batch_labels,
                comb_emb_txt,
                epoch,
            )
            loss_txt = loss_dict_txt["total_loss"]
            batch_log_dict_txt = {
                f"train2_img/{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict_txt.items()
            }
            logger.log_metrics(batch_log_dict_txt)

            comb_emb_origin = model.combine(
                txt_features,
                None,
                label_embeddings,
                epoch=epoch,
                return_label_proj=False,
            )

            # Add two monitors of how much changed in the condition
            diff_img = torch.nn.functional.mse_loss(comb_emb_img, comb_emb_origin)
            diff_txt = torch.nn.functional.mse_loss(comb_emb_txt, comb_emb_origin)
            # diff_imgtxt = torch.nn.functional.mse_loss(comb_emb_both, comb_emb_origin)

            # Check magnitude
            magnitude_img = torch.norm(pred_cond_img, p=2, dim=1).mean()
            magnitude_txt = torch.norm(pred_cond_txt, p=2, dim=1).mean()
            # magnitude_imgtxt = torch.norm(pred_cond_both, p=2, dim=1).mean()

            # Check diversity
            diversity_img = pred_cond_img.std(dim=0).mean()
            diversity_txt = pred_cond_txt.std(dim=0).mean()
            # diversity_imgtxt = pred_cond_both.std(dim=0).mean()

            # Check seperation
            # sim_both = (
            #     torch.nn.functional.normalize(comb_emb_both, p=2, dim=1)
            #     @ torch.nn.functional.normalize(comb_emb_origin, p=2, dim=1).T
            # )
            sim_img = (
                torch.nn.functional.normalize(comb_emb_img, p=2, dim=1)
                @ torch.nn.functional.normalize(comb_emb_origin, p=2, dim=1).T
            )
            sim_txt = (
                torch.nn.functional.normalize(comb_emb_txt, p=2, dim=1)
                @ torch.nn.functional.normalize(comb_emb_origin, p=2, dim=1).T
            )

            # pos_sims_both = sim_both.diagonal()
            pos_sims_img = sim_img.diagonal()
            pos_sims_txt = sim_txt.diagonal()

            # neg_sims_both = sim_both[~torch.eye(len(sim_both), dtype=torch.bool)]
            neg_sims_img = sim_img[~torch.eye(len(sim_img), dtype=torch.bool)]
            neg_sims_txt = sim_txt[~torch.eye(len(sim_txt), dtype=torch.bool)]

            seperation_dict = {
                # "pos_mean_both": pos_sims_both.mean(),
                # "pos_std_both": pos_sims_both.std(),
                # "neg_mean_both": neg_sims_both.mean(),
                # "neg_std_both": neg_sims_both.std(),
                "pos_mean_img": pos_sims_img.mean(),
                "pos_std_img": pos_sims_img.std(),
                "neg_mean_img": neg_sims_img.mean(),
                "neg_std_img": neg_sims_img.std(),
                "pos_mean_txt": pos_sims_txt.mean(),
                "pos_std_txt": pos_sims_txt.std(),
                "neg_mean_txt": neg_sims_txt.mean(),
                "neg_std_txt": neg_sims_txt.std(),
            }

            batch_log_dict_seperation = {
                f"train_seperation/{k}": v.item() if torch.is_tensor(v) else v
                for k, v in seperation_dict.items()
            }
            logger.log_metrics(batch_log_dict_seperation)

            loss_dict = {
                # "loss_both": loss_both,
                "loss_img": loss_img,
                "loss_txt": loss_txt,
                "loss_total": loss_img + loss_txt,
            }

            loss = loss_dict["loss_total"]

            epoch_loss += loss.item()
            num_batches += 1

            # Log batch-level loss components
            batch_log_dict = {
                f"train_2/batch_{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
            }
            batch_log_dict.update(
                {
                    "train_2_other/epoch": epoch,
                    "train_2_other/batch": batch_idx,
                    "train_2_other/step": global_step,
                    "train_3/diff_embedding_img": (diff_img.item()),
                    "train_3/diff_embedding_txt": (diff_txt.item()),
                    # "train_3/diff_embedding_imgtxt": (diff_imgtxt.item()),
                    "train_3/magnitude_img": (magnitude_img.item()),
                    "train_3/magnitude_txt": (magnitude_txt.item()),
                    # "train_3/magnitude_imgtxt": (magnitude_imgtxt.item()),
                    "train_3/diversity_img": (diversity_img.item()),
                    "train_3/diversity_txt": (diversity_txt.item()),
                    # "train_3/diversity_imgtxt": (diversity_imgtxt.item()),
                }
            )
            logger.log_metrics(batch_log_dict)

            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()  # Update model parameters with accumulated state
            global_step += 1

        # ========== EPOCH END: Performance Monitoring & Sync ==========
        epoch_time = time.time() - epoch_start_time

        # Log epoch performance
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

        # Log to wandb logger
        logger.log_metrics(
            {
                "train_2_other/epoch": epoch,
                "train_2/loss": avg_loss,
                "train_2_other/epoch_time": epoch_time,
            }
        )

        scheduler_2.step()

        # Test evaluation, this time we test for every epoch
        with torch.no_grad():
            # Test evaluation
            test_results_detailed = evaluator.evaluate_test(
                model=model,
                processor=processor,
                dataloader=test_loader,
                label_embeddings=None,  # Your label embeddings
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

            # Log test results
            for metric, value in test_results.metrics.items():
                logger.log_metrics({metric: value})

    for metric, value in test_results.metrics.items():
        logger.log_metrics({f"Test2_final_{metric}": value})

    # Visualize condition predictions
    print("Visualizing condition predictions")
    with torch.no_grad():
        model.eval()
        all_img_condition_predicted = []
        all_txt_condition_predicted = []
        all_imgtxt_condition_predicted = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Use batch_idx as chunk_id for direct chunk loading
            chunk_id = batch_idx

            # Load features by chunk directly
            features_data = feature_manager.get_features_by_chunk(chunk_id)

            img_features = features_data["img_features"].to(device, non_blocking=True)
            txt_features = features_data["txt_features"].to(device, non_blocking=True)

            # Get embeddings for this shard by sample_id
            batch_sample_ids = features_data["sample_ids"].tolist()
            label_embeddings = embedding_manager.get_embeddings(batch_sample_ids).to(device)

            # predict condition
            img_condition_predicted = model.predict_condition(
                img_features,
                None,
                "txt",
                return_logits=False,
            )  # Here we predict the condition, which is the opposite of the mask type
            txt_condition_predicted = model.predict_condition(
                None,
                txt_features,
                "img",
                return_logits=False,
            )
            imgtxt_condition_predicted = model.predict_condition(
                img_features,
                txt_features,
                "imgtxt",
                return_logits=False,
            )
            all_img_condition_predicted.append(img_condition_predicted)
            all_txt_condition_predicted.append(txt_condition_predicted)
            all_imgtxt_condition_predicted.append(imgtxt_condition_predicted)

        all_img_condition_predicted = torch.cat(all_img_condition_predicted, dim=0)
        all_txt_condition_predicted = torch.cat(all_txt_condition_predicted, dim=0)
        all_imgtxt_condition_predicted = torch.cat(
            all_imgtxt_condition_predicted, dim=0
        )

        all_img_condition_predicted = all_img_condition_predicted.cpu().numpy()
        all_txt_condition_predicted = all_txt_condition_predicted.cpu().numpy()
        all_imgtxt_condition_predicted = all_imgtxt_condition_predicted.cpu().numpy()

        # Visualize test results
        _, label_embeddings_all = embedding_manager.get_all_embeddings()
        umap_vis = UMAP_vis(device=device)
        # # Check whether label embeddings are 2d or higher dimensional
        if label_embeddings_all.shape[1] == 2:
            umap_features = label_embeddings_all.cpu().numpy()
        else:
            umap_features = umap_vis.learn_umap(
                label_embeddings_all, close_cluster=True
            )

        full_list = {
            "img_condition_predicted": all_img_condition_predicted,
            "txt_condition_predicted": all_txt_condition_predicted,
            "imgtxt_condition_predicted": all_imgtxt_condition_predicted,
            "label_embeddings": label_embeddings_all,
        }

        for key, value in full_list.items():
            # Visualize label embeddings
            fig = get_umap(
                value,
                umap_labels=None,
                epoch=epoch,
                no_outlier=True,
                samples_to_track=[0, 1, 2, 3, 4],
            )
            experiment.save_artifact(
                name=f"label_embeddings_umap_{key}",
                data=fig,
                artifact_type="figure",
                folder="plots",
                description=f"UMAP visualization of {key}",
            )

            fig2 = visualize_ideal_condition_space(value, epoch)
            experiment.save_artifact(
                name=f"ideal_condition_space_{key}",
                data=fig2,
                artifact_type="figure",
                folder="plots",
                description=f"Ideal condition space visualization of {key}",
            )

            logger.log_metrics(
                {
                    f"vis/umap_{key}": wandb.Image(fig),
                    f"vis/ideal_condition_space_{key}": wandb.Image(fig2),
                }
            )

            plt.close("all")

    experiment.save_artifact(
        name="final_model",
        folder="checkpoints",
        data=model.combiner.state_dict(),
        artifact_type="torch",
        description="Final model combiner state dictionary",
    )

    print("Training Complete!")

    return 0
