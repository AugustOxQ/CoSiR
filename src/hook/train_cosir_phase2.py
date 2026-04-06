"""
This file is used to take checkpoint from phase 1 and train the condition predictor only for phase 2.
"""

import time
from pathlib import Path
import os
import torch
import torch.nn.functional as F
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
)
from src.model import CoSiRModel, Clustering, UMAP_vis
from src.eval import EvaluationManager, EvaluationConfig
from src.utils import (
    FeatureManager,
    ExperimentManager,
    TrainableEmbeddingManager,
    visualize_angular_semantics_fast,
    visualize_angular_semantics_text_to_image_fast,
    get_representatives_polar_grid,
    get_representatives_hdbscan,
    get_umap,
    visualize_ideal_condition_space,
)
from src.metrics import (
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

    # Set phase_1 overall root path
    phase_1_root_path = experiment.directory.parent
    phase_1_root_path = Path(phase_1_root_path / "20260401_032751_CoSiR_Experiment")

    # Load phase 1 model
    _phase_1_model_path = str(phase_1_root_path / "checkpoints")

    _phase_1_model_files = sorted(Path(_phase_1_model_path).glob("phase_1_model*.pt"))
    if not _phase_1_model_files:
        raise FileNotFoundError(f"No phase_1_model*.pkl found in {_phase_1_model_path}")
    with open(_phase_1_model_files[0], "rb") as f:
        phase_1_model_path = _phase_1_model_files[0]

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

    # Initialize evaluator
    evaluator = EvaluationManager(evaluation_config)

    clustering = Clustering(device=device)

    # ========== OPTIMIZED: TrainableEmbeddingManager with Intelligent Caching ==========

    # Auto-optimize chunk size based on batch size if not specified
    embedding_chunk_size = cfg.embeddingmanager.embedding_chunk_size
    if embedding_chunk_size is None:
        # Optimal chunk size: 4x batch size, capped at reasonable limits
        embedding_chunk_size = cfg.featuremanager.chunk_size
        print(
            f"Auto-optimized embedding chunk size: {embedding_chunk_size} (4x batch_size)"
        )

    sample_ids_list_2_path = "/data/SSD2/pre_extract/impressions/sample_ids.pt"
    print(f"Loading sample ids list 2 from {sample_ids_list_2_path}")
    sample_ids_list_2 = torch.load(sample_ids_list_2_path)

    embedding_manager = TrainableEmbeddingManager(
        sample_ids=sample_ids_list_2,  # Use actual sample IDs from feature extraction
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
    phase_1_embeddings_path = str(phase_1_root_path / "final_embeddings")
    phase_1_other_path = str(phase_1_root_path / "embeddings")

    embedding_manager.load_phase_1_template(phase_1_embeddings_path)

    # Load other important information from phase 1, pkl format
    # Use prefix matching to support dated filenames like id_to_chunk_index_20250301.pkl
    _other_dir = Path(phase_1_other_path)
    _id_to_chunk_files = sorted(_other_dir.glob("id_to_chunk_index*.pkl"))
    if not _id_to_chunk_files:
        raise FileNotFoundError(
            f"No id_to_chunk_index*.pkl found in {phase_1_other_path}"
        )
    with open(_id_to_chunk_files[0], "rb") as f:
        id_to_chunk_index = pickle.load(f)  # type: ignore
    _chunk_mapping_files = sorted(_other_dir.glob("chunk_mapping*.pkl"))
    if not _chunk_mapping_files:
        raise FileNotFoundError(f"No chunk_mapping*.pkl found in {phase_1_other_path}")
    with open(_chunk_mapping_files[0], "rb") as f:
        chunk_mapping = pickle.load(f)  # type: ignore

    embedding_manager.id_to_chunk_index = id_to_chunk_index
    embedding_manager.chunk_mapping = chunk_mapping

    # Optimize cache settings based on actual batch size
    embedding_manager.optimize_cache_settings(cfg.featuremanager.chunk_size)

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
        ratio=(
            1.0 if cfg.data.dataset_type != "redcaps2" else 0.2
        ),  # TODO: change to 0.1 for redcaps2
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
    print("Getting all embeddings")
    _, label_embeddings_all = embedding_manager.get_all_embeddings()

    print("Getting HDBSCAN labels")
    hdbscanlabels, _ = clustering.get_hdbscan(label_embeddings_all.cpu(), method="leaf")

    print("Getting pre-trained representatives")

    print("Getting representatives")
    pre_trained_representatives = get_representatives_hdbscan(
        hdbscanlabels, label_embeddings_all.cpu(), cfg.train.representative_number
    )

    pre_trained_representatives_device = pre_trained_representatives.to(device)

    model.set_pretrained_representatives(pre_trained_representatives, device)

    # print("Evaluating test results with oracle")
    # with torch.no_grad():
    #     test_results_detailed = evaluator.evaluate_test(
    #         model=model,
    #         processor=processor,
    #         dataloader=test_loader,
    #         label_embeddings=pre_trained_representatives_device,  # Your label embeddings
    #         epoch=0,
    #         return_detailed_results=True,  # 记住这里是true，也就是detailed_results
    #         use_oracle=True,
    #     )

    #     (
    #         all_img_emb,
    #         all_txt_emb,
    #         all_raw_text,
    #         text_to_image_map,
    #         image_to_text_map,
    #         test_results,
    #     ) = test_results_detailed  # type: ignore

    #     # Log test results
    #     for metric, value in test_results.metrics.items():
    #         logger.log_metrics({f"Test2_oracle_{metric}": value})

    #     all_raw_image = test_set.get_all_raw_image()

    #     # Log test results
    #     for metric, value in test_results.metrics.items():
    #         logger.log_metrics({metric: value})

    #     # Visualize test results
    #     # # Check whether label embeddings are 2d or higher dimensional
    #     if label_embeddings_all.shape[1] == 2:
    #         umap_features = label_embeddings_all.cpu().numpy()
    #     else:
    #         umap_features = umap_vis.learn_umap(
    #             label_embeddings_all, close_cluster=True
    #         )

    #     # Visualize label embeddings
    #     fig = get_umap(
    #         umap_features,
    #         umap_labels=None,
    #         epoch=0,
    #         no_outlier=True,
    #         samples_to_track=[0, 1, 2, 3, 4],
    #     )
    #     experiment.save_artifact(
    #         name=f"label_embeddings_umap_0",
    #         data=fig,
    #         artifact_type="figure",
    #         folder="plots",
    #         description=f"UMAP visualization of trained label embeddings at epoch 0",
    #     )

    #     fig2 = visualize_ideal_condition_space(umap_features, 0)
    #     experiment.save_artifact(
    #         name=f"ideal_condition_space_0",
    #         data=fig2,
    #         artifact_type="figure",
    #         folder="plots",
    #         description=f"Ideal condition space visualization at epoch 0",
    #     )

    #     logger.log_metrics(
    #         {
    #             "vis/umap": wandb.Image(fig),
    #             "vis/ideal_condition_space": wandb.Image(fig2),
    #         }
    #     )

    #     plt.close("all")

    #     for tmp_round in range(6):
    #         fig3 = visualize_angular_semantics_fast(
    #             label_embeddings_all.cpu().numpy(),
    #             model,
    #             (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
    #             device=device,
    #         )
    #         experiment.save_artifact(
    #             name=f"angular_semantics_fast_0_tmp_{tmp_round}",
    #             data=fig3,
    #             artifact_type="figure",
    #             folder="plots",
    #             description=f"Angular semantics visualization at epoch 0",
    #         )

    #         fig4 = visualize_angular_semantics_text_to_image_fast(
    #             label_embeddings_all.cpu().numpy(),
    #             model,
    #             (
    #                 all_img_emb,
    #                 all_txt_emb,
    #                 all_raw_image,
    #                 all_raw_text,
    #             ),
    #             device=device,
    #         )

    #         experiment.save_artifact(
    #             name=f"angular_semantics_text_to_image_0_tmp_{tmp_round}",
    #             data=fig4,
    #             artifact_type="figure",
    #             folder="plots",
    #             description=f"Angular semantics visualization (text-to-image) at epoch 0",
    #         )

    #         plt.close("all")

    #     cosir_automatic_evaluator = CoSiRAutomaticEvaluator(
    #         model,
    #         (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
    #         label_embeddings_all,
    #         device,
    #     )
    #     result = cosir_automatic_evaluator.evaluate_all()
    #     logger.log_metrics(result)

    print("Pre-computing ground truth embeddings")

    ground_truth_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Use batch_idx as chunk_id for direct chunk loading
            chunk_id = batch_idx

            # Load features by chunk directly
            features_data = feature_manager.get_features_by_chunk(chunk_id)

            img_features = features_data["img_features"].to(device, non_blocking=True)
            txt_features = features_data["txt_features"].to(device, non_blocking=True)

            batch_size = len(img_features)
            # best rank seen so far per sample (0 = perfect, i.e. diagonal is top-1)
            best_ranks = torch.full((batch_size,), float("inf"), device=device)
            # which rep index produced that best rank (default: rep 0)
            batch_ground_truth = torch.zeros(batch_size, dtype=torch.long)

            img_norm = F.normalize(img_features, dim=1)

            for rep_idx, rep in enumerate(pre_trained_representatives_device):
                # rep: [label_dim] -> expand to [B, label_dim] for the combiner
                rep_expanded = rep.unsqueeze(0).expand(batch_size, -1)
                batch_combined = model.combine(
                    txt_features,
                    None,
                    rep_expanded,
                    epoch=0,
                    return_label_proj=False,
                )
                # full cosine-sim matrix [B, B] — considers off-diagonal negatives
                combined_norm = F.normalize(batch_combined, dim=1)
                sim_matrix = combined_norm @ img_norm.T  # [B, B]
                diag_sim = sim_matrix.diagonal()  # [B]
                # 0-indexed rank of the correct match within its row (0 = ranked #1)
                ranks = (sim_matrix > diag_sim.unsqueeze(1)).sum(dim=1).float()

                # update ground truth where this rep gives a better rank
                update_mask = ranks < best_ranks
                batch_ground_truth[update_mask] = rep_idx
                best_ranks[update_mask] = ranks[update_mask]

            ground_truth_list.append(batch_ground_truth)

    # Start label prediction loss training
    print("Starting label prediction loss training")
    epoch = 0
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

            # # Get embeddings by chunk directly from optimized manager
            # chunk_sample_ids, label_embeddings_data = (
            #     embedding_manager.get_embeddings_by_chunk(chunk_id)
            # )
            # label_embeddings = label_embeddings_data.to(device)

            batch_labels = ground_truth_list[batch_idx].to(device)

            # Start both prediction
            pred_cond_both, logits_both = model.predict_condition(
                img_features,
                txt_features,
                "imgtxt",
                training_phase=True,
            )
            comb_emb_both = model.combine(
                txt_features,
                None,
                pred_cond_both,
                epoch=epoch,
                return_label_proj=False,
            )
            loss_dict_both = criteria_2(
                logits_both,
                pred_cond_both,
                batch_labels,
                epoch,
            )
            loss_both = loss_dict_both["total_loss"]
            batch_log_dict_both = {
                f"train2_both/{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict_both.items()
            }
            logger.log_metrics(batch_log_dict_both)

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
                batch_labels,
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
                batch_labels,
                epoch,
            )
            loss_txt = loss_dict_txt["total_loss"]
            batch_log_dict_txt = {
                f"train2_img/{k}": v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict_txt.items()
            }
            logger.log_metrics(batch_log_dict_txt)

            # Check magnitude
            magnitude_img = torch.norm(pred_cond_img, p=2, dim=1).mean()
            magnitude_txt = torch.norm(pred_cond_txt, p=2, dim=1).mean()
            magnitude_imgtxt = torch.norm(pred_cond_both, p=2, dim=1).mean()

            # Check diversity
            diversity_img = pred_cond_img.std(dim=0).mean()
            diversity_txt = pred_cond_txt.std(dim=0).mean()
            diversity_imgtxt = pred_cond_both.std(dim=0).mean()

            loss_dict = {
                "loss_both": loss_both,
                "loss_img": loss_img,
                "loss_txt": loss_txt,
                "loss_total": loss_both + loss_img + loss_txt,
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
                    "train_3/magnitude_img": (magnitude_img.item()),
                    "train_3/magnitude_txt": (magnitude_txt.item()),
                    "train_3/magnitude_imgtxt": (magnitude_imgtxt.item()),
                    "train_3/diversity_img": (diversity_img.item()),
                    "train_3/diversity_txt": (diversity_txt.item()),
                    "train_3/diversity_imgtxt": (diversity_imgtxt.item()),
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

        if (
            epoch == 0
            or epoch % cfg.eval.evaluation_interval == 0
            or epoch == cfg.train.epochs_2 - 1
        ):
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
        all_img_features = []
        all_txt_features = []
        ground_truth_labels = []
        all_img_condition_predicted = []
        all_txt_condition_predicted = []
        all_imgtxt_condition_predicted = []
        all_combined_condition_img = []
        all_combined_condition_txt = []
        all_combined_condition_imgtxt = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):

            if batch_idx >= 20:
                break

            # Use batch_idx as chunk_id for direct chunk loading
            chunk_id = batch_idx

            # Load features by chunk directly
            features_data = feature_manager.get_features_by_chunk(chunk_id)

            img_features = features_data["img_features"].to(device, non_blocking=True)
            txt_features = features_data["txt_features"].to(device, non_blocking=True)

            batch_labels = ground_truth_list[batch_idx].to(device)

            # predict condition
            # Img
            img_condition_predicted = model.predict_condition(
                img_features,
                None,
                "txt",
                return_logits=False,
            )  # Here we predict the condition, which is the opposite of the mask type
            combined_condition_img = model.combine(
                txt_features,
                None,
                img_condition_predicted,
                epoch=epoch,
                return_label_proj=False,
            )
            txt_condition_predicted = model.predict_condition(
                None,
                txt_features,
                "img",
                return_logits=False,
            )
            combined_condition_txt = model.combine(
                txt_features,
                None,
                txt_condition_predicted,
                epoch=epoch,
                return_label_proj=False,
            )
            imgtxt_condition_predicted = model.predict_condition(
                img_features,
                txt_features,
                "imgtxt",
                return_logits=False,
            )
            combined_condition_imgtxt = model.combine(
                txt_features,
                None,
                imgtxt_condition_predicted,
                epoch=epoch,
                return_label_proj=False,
            )
            all_img_condition_predicted.append(img_condition_predicted)
            all_txt_condition_predicted.append(txt_condition_predicted)
            ground_truth_labels.append(batch_labels)
            all_imgtxt_condition_predicted.append(imgtxt_condition_predicted)
            all_img_features.append(img_features)
            all_txt_features.append(txt_features)
            all_combined_condition_img.append(combined_condition_img)
            all_combined_condition_txt.append(combined_condition_txt)
            all_combined_condition_imgtxt.append(combined_condition_imgtxt)

        all_img_condition_predicted = torch.cat(all_img_condition_predicted, dim=0)
        all_txt_condition_predicted = torch.cat(all_txt_condition_predicted, dim=0)
        all_imgtxt_condition_predicted = torch.cat(
            all_imgtxt_condition_predicted, dim=0
        )
        ground_truth_labels = torch.cat(ground_truth_labels, dim=0)

        all_img_condition_predicted = all_img_condition_predicted.cpu().numpy()
        all_txt_condition_predicted = all_txt_condition_predicted.cpu().numpy()
        all_imgtxt_condition_predicted = all_imgtxt_condition_predicted.cpu().numpy()
        ground_truth_labels = pre_trained_representatives_device[ground_truth_labels]

        # count the number of each label
        ground_truth_labels, label_counts = torch.unique(
            ground_truth_labels, return_counts=True, dim=0
        )
        # Sort the labels by count
        ground_truth_labels = ground_truth_labels[label_counts.argsort(descending=True)]
        label_counts = label_counts[label_counts.argsort(descending=True)]
        ground_truth_labels = ground_truth_labels.cpu().numpy()
        label_counts = label_counts.cpu().numpy()
        # Print each label and its count
        print("Label counts:")
        for label, count in zip(ground_truth_labels[:10], label_counts[:10]):
            print(f"Label {label}: {count} samples")
        # Print the total number of samples
        print(f"Total number of samples: {len(ground_truth_labels)}")

        # Visualize test results
        _, label_embeddings_all = embedding_manager.get_all_embeddings()

        # plot ground_truth_labels, where the size of the point represents the count of the label
        plt.figure(figsize=(10, 10))
        max_size, min_size = 500, 10
        log_counts = np.log(label_counts)
        size = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min())
        size = size * (max_size - min_size) + min_size
        plt.scatter(
            ground_truth_labels[:, 0],
            ground_truth_labels[:, 1],
            s=size,
        )

        experiment.save_artifact(
            name=f"ground_truth_labels",
            data=plt.gcf(),
            artifact_type="figure",
            folder="plots",
            description=f"Ground truth labels visualization",
        )

        logger.log_metrics(
            {
                f"vis/ground_truth_labels": wandb.Image(plt.gcf()),
            }
        )
        plt.close()

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

        all_img_features = torch.cat(all_img_features, dim=0)
        all_txt_features = torch.cat(all_txt_features, dim=0)
        all_combined_condition_img = torch.cat(all_combined_condition_img, dim=0)
        all_combined_condition_txt = torch.cat(all_combined_condition_txt, dim=0)
        all_combined_condition_imgtxt = torch.cat(all_combined_condition_imgtxt, dim=0)

        all_img_features_umap = umap_vis.learn_umap(
            all_img_features, close_cluster=True
        )
        all_txt_features_umap = umap_vis.learn_umap(
            all_txt_features, close_cluster=True
        )
        all_combined_condition_img_umap = umap_vis.learn_umap(
            all_combined_condition_img, close_cluster=True
        )
        all_combined_condition_txt_umap = umap_vis.learn_umap(
            all_combined_condition_txt, close_cluster=True
        )
        all_combined_condition_imgtxt_umap = umap_vis.learn_umap(
            all_combined_condition_imgtxt, close_cluster=True
        )

        full_list_combined = {
            "img_features": all_img_features_umap,
            "txt_features": all_txt_features_umap,
            "combined_condition_img": all_combined_condition_img_umap,
            "combined_condition_txt": all_combined_condition_txt_umap,
            "combined_condition_imgtxt": all_combined_condition_imgtxt_umap,
        }

        for key, value in full_list_combined.items():
            # Visualize label embeddings
            fig = get_umap(
                value,
                umap_labels=None,
                epoch=epoch,
                no_outlier=True,
                samples_to_track=[0, 1, 2, 3, 4],
            )
            experiment.save_artifact(
                name=f"Check_change_{key}",
                data=fig,
                artifact_type="figure",
                folder="plots",
                description=f"UMAP visualization of the change between {key} and the origin",
            )

            logger.log_metrics(
                {
                    f"vis2/umap_{key}": wandb.Image(fig),
                }
            )

            plt.close("all")

    experiment.save_artifact(
        name="full_list",
        data=full_list,
        artifact_type="pickle",
        description="Full list of conditions",
        folder="embeddings",
    )

    experiment.save_artifact(
        name="full_list_combined",
        data=full_list_combined,
        artifact_type="pickle",
        description="Full list of combined features",
        folder="embeddings",
    )

    experiment.save_artifact(
        name="final_model",
        folder="checkpoints",
        data=model.combiner.state_dict(),
        artifact_type="torch",
        description="Final model combiner state dictionary",
    )
    print("Training Complete!")

    return 0
