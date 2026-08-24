"""
CoSiR phase-1 training loop.

Entry point: train_cosir(cfg, logger)
Private helpers (prefix _) own each setup phase so the main loop stays readable.
To add a new setup phase, write a new _<phase> function and call it from train_cosir().
"""
import time
import os
import pathlib
import json
import torch
import torch.nn.functional as F
import random
from typing import cast, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
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

from src.dataset import (
    CoSiRShardDataset,
    CoSiRShardStreamDataset,
    CoSiRValidationDataset,
    FeatureExtractionDataset,
    FeatureExtractionConceptualDataset,
)
from src.model import CoSiRModel, UMAP_vis
from src.eval import EvaluationManager, EvaluationConfig, TestEvaluationDetail
from src.utils import (
    FeatureManager,
    ExperimentManager,
    TrainableEmbeddingManager,
    TemplateIncompatibleError,
    get_representatives_fps,
    get_umap,
    visualize_ideal_condition_space,
    CoSiRAutomaticEvaluator,
)
from src.metrics import LabelContrastiveLoss_enhance
from src.metrics.regularizer import (
    build_neighbor_csr,
    buddy_graph_smoothness_loss,
    buddy_contrastive_loss,
    reorder_features_to_z,
    refresh_buddy_graph,
    edge_jaccard,
    compute_comb_all_eval,
    buddy_knn_preservation,
)


# ─── Setup helpers ──────────────────────────────────────────────────────────
# Each helper owns exactly one setup concern and returns only what its caller needs.
# To add a new setup phase, define a new _<phase> function and call it from train_cosir().


def _setup_model_and_criteria(cfg, device):
    """Return (EvaluationConfig, CoSiRModel, CLIP processor, loss criteria)."""
    evaluation_config = EvaluationConfig(
        device=device,
        k_vals=cfg.eval.k_vals,
        train_max_batches=cfg.eval.train_max_batches,
        print_metrics=cfg.eval.print_metrics,
        evaluation_interval=(
            cfg.eval.evaluation_interval if cfg.eval.evaluation_interval > 0 else 5
        ),
    )

    print("Initializing model")
    model = CoSiRModel(
        backbone_model=cfg.model.clip_model,
        label_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.hidden_dim,
        num_conditions=cfg.train.representative_number,
        dropout=cfg.model.dropout,
        combine_side=cfg.model.combine_side,
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.model.clip_model, use_fast=False)

    print("Initializing criteria")
    criteria = LabelContrastiveLoss_enhance(
        margin=cfg.loss.margin,
        lambda_contrastive=cfg.loss.lambda_contrastive,
        lambda_laplacian=cfg.loss.lambda_laplacian,
        lambda_collapse=cfg.loss.lambda_collapse,
        lambda_boundary=cfg.loss.lambda_boundary,
        lambda_mixup=cfg.loss.lambda_mixup,
        lambda_delta=cfg.loss.lambda_delta,
        lambda_gate=cfg.loss.lambda_gate,
        lambda_gate_logit=cfg.loss.lambda_gate_logit,
        lambda_preserve=cfg.loss.lambda_preserve,
        mixup_alpha=cfg.loss.mixup_alpha,
        return_dict=cfg.loss.return_dict,
    )
    return evaluation_config, model, processor, criteria


def _extract_or_load_features(cfg, model, processor, device):
    """Load cached CLIP features from disk, or extract and cache them from the dataset.

    Returns (FeatureManager, sample_ids_list).
    """
    storage_dir = cfg.featuremanager.storage_dir
    feature_manager = FeatureManager(
        storage_dir,
        shard_size=cfg.featuremanager.shard_size,
        hdf5_compression=cfg.featuremanager.hdf5_compression,
        hdf5_compression_level=cfg.featuremanager.hdf5_compression_level,
    )

    metadata_path = os.path.join(storage_dir, "metadata.json")
    if os.path.exists(metadata_path) and cfg.train.load_existing_features:
        print("Loading existing feature store")
        feature_manager.validate_backbone(cfg.model.clip_model)
        sample_ids_list = feature_manager.get_all_sample_ids()
        print(f"Loaded {len(sample_ids_list):,} sample ids from existing store")
        return feature_manager, sample_ids_list

    print("Extracting features")
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

            img_e, txt_e, img_f, txt_f = model.encode_img_txt(image_inputs, text_inputs)
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
    return feature_manager, sample_ids_list


def _setup_experiment(cfg, evaluation_config, device, logger):
    """Create experiment directory, test evaluator, and UMAP projector.

    Returns (ExperimentContext, EvaluationManager, UMAP_vis).
    """
    exp_manager = ExperimentManager(cfg.experiment.results_dir)
    experiment = exp_manager.create_experiment(
        name=cfg.experiment.name,
        config=cfg,
        tags=cfg.experiment.tags,
        description=cfg.experiment.description,
    )
    print(f"Created experiment: {experiment.name}")
    print(f"Experiment directory: {experiment.directory}")

    evaluator = EvaluationManager(evaluation_config)
    # Point the evaluator's cache at the experiment directory so frozen backbone
    # embeddings are saved after the first extraction and reused across epochs.
    evaluator.config.cache_dir = experiment.directory
    umap_vis = UMAP_vis(device=device)
    return experiment, evaluator, umap_vis


def _init_embedding_manager(cfg, device, sample_ids_list, experiment, feature_manager, model):
    """Initialize TrainableEmbeddingManager and run the configured initialization strategy.

    Tries to reuse a cached template; falls back to fresh initialization.
    Returns the initialized embedding_manager.
    """
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

    strategy = cfg.train.initialization_strategy
    if strategy == "buddy":  # accept the singular spelling as an alias for "buddies"
        strategy = "buddies"
    if strategy not in ("imgtxt", "txt", "img", "buddies"):
        return embedding_manager

    # Buddy hyperparameters (only meaningful for strategy == "buddies"); the
    # 'extra' dict feeds the template-compatibility guard so a stale buddies
    # template is rejected when k / alpha / method change.
    _bud = getattr(cfg.train, "buddies", None)
    _buddy_kwargs = {
        "k": int(getattr(_bud, "k", 30)) if _bud is not None else 30,
        "alpha": float(getattr(_bud, "alpha", 0.5)) if _bud is not None else 0.5,
        "method": str(getattr(_bud, "method", "spectral")) if _bud is not None else "spectral",
        "knn_batch_size": int(getattr(_bud, "knn_batch_size", 1024)) if _bud is not None else 1024,
        "normalize_method": str(getattr(_bud, "normalize_method", "rank")) if _bud is not None else "rank",
        "seed": int(cfg.seed),
        "b_weight": float(getattr(_bud, "b_weight", 1.0)) if _bud is not None else 1.0,
        "distance_mode": str(getattr(_bud, "distance_mode", "blend")) if _bud is not None else "blend",
    }
    # Experiment 8 (buddy-init encoder-pair ablation, docs/superpowers/specs/
    # 2026-08-04-buddy-publication-plan-design.md §4): swap which (vision, text) encoder
    # pair BUILDS the buddy graph/init, while the frozen CLIP training backbone stays
    # untouched. Set via `+train.buddies.encoder_pair=<vision>:<text>` (e.g. "dinov2:bge");
    # absent by default, which preserves the exact original CLIP-FeatureManager code path.
    _encoder_pair = getattr(_bud, "encoder_pair", None) if _bud is not None else None
    if _encoder_pair:
        from src.conditional_buddy.heldout_encoder_features import load_encoder_pair_features
        _vision, _text = str(_encoder_pair).split(":")
        _img_ov, _txt_ov, _ids_ov = load_encoder_pair_features(
            cfg.data.dataset_type, _vision, _text, feature_manager
        )
        _buddy_kwargs["feature_override"] = (_img_ov, _txt_ov, _ids_ov)
    _extra = None
    if strategy == "buddies":
        _extra = {"k": _buddy_kwargs["k"], "alpha": _buddy_kwargs["alpha"],
                  "method": _buddy_kwargs["method"]}
        # Only add b_weight to the template key when it departs from the default, so
        # existing (pre-b_weight) templates stay compatible for standard runs while a
        # changed lean still forces a rebuild (no silent template reuse across values).
        if _buddy_kwargs["b_weight"] != 1.0:
            _extra["b_weight"] = _buddy_kwargs["b_weight"]
        if _encoder_pair:
            _extra["encoder_pair"] = _encoder_pair
        # Only add distance_mode to the template key when it departs from the default,
        # so existing (pre-distance_mode) templates stay compatible for standard runs
        # while a changed mode still forces a rebuild (no silent template reuse across
        # blend/typed).
        if _buddy_kwargs["distance_mode"] != "blend":
            _extra["distance_mode"] = _buddy_kwargs["distance_mode"]

    template_dir = experiment.directory.parent / "template_embeddings"
    template_exists = template_dir.exists() and (template_dir / "embeddings.npy").exists()

    _need_initialize = True
    _need_save_template = getattr(cfg.train, "save_as_template_embeddings", True)

    if template_exists and getattr(cfg.train, "use_template_embeddings", True):
        print("Attempting to load from template embeddings...")
        try:
            embedding_manager.load_imgtxt_template(
                strategy=strategy,
                factor=cfg.train.imgtxt_factor,
                normalize=cfg.train.normalize,
                extra=_extra,
            )
            _need_initialize = False
            _need_save_template = False
        except TemplateIncompatibleError as e:
            print(f"Template config mismatch: {e}")
            print("Re-initializing and overwriting template with current config...")
        except Exception as e:
            print(f"Failed to load template: {e}")
            print("Falling back to initialization (template will not be overwritten)...")
            _need_save_template = False

    if _need_initialize:
        print(f"Initializing embeddings with {strategy} strategy...")
        if strategy == "buddies":
            embedding_manager.initialize_embeddings_buddies(
                feature_manager, model, device, **_buddy_kwargs
            )
        else:
            _init_fn_map = {
                "imgtxt": embedding_manager.initialize_embeddings_imgtxt,
                "txt": embedding_manager.initialize_embeddings_txt,
                "img": embedding_manager.initialize_embeddings_img,
            }
            _init_fn_map[strategy](
                feature_manager,
                model,
                device,
                factor=cfg.train.imgtxt_factor,
                normalize=cfg.train.normalize,
            )

        if _need_save_template:
            print("Storing embeddings as template for future use...")
            embedding_manager.store_imgtxt_template(
                strategy=strategy,
                factor=cfg.train.imgtxt_factor,
                normalize=cfg.train.normalize,
                extra=_extra,
            )

    return embedding_manager


def _build_optimizer_and_scheduler(cfg, model, embedding_manager):
    """Return (optimizer, scheduler) built after embedding initialization.

    Must be called after _init_embedding_manager so embedding_manager.embeddings
    is the final nn.Parameter (template loading may replace it).
    """
    print("Initializing optimizer and scheduler")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "condition_predictor" not in n and "other_proj" not in n
                ],
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": [embedding_manager.embeddings],
                "lr": cfg.optimizer.lr_label,
                "weight_decay": 0,
            },
            {
                "params": list(model.condition_predictor.parameters()),
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": list(model.other_proj.parameters()),
                "lr": cfg.optimizer.lr / 10,
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

    return optimizer, scheduler


def _build_dataloaders(cfg, feature_manager, processor, sample_ids_list):
    """Build train and test DataLoaders; also return per-sample type array (Impressions only).

    Returns (train_set, train_loader, test_set, test_loader, sample_types).
    sample_types is a numpy int array for Impressions, otherwise an empty list.
    """
    feature_types = feature_manager.available_features  # e.g. ['img_features','txt_features']

    # ── Sample types (must be computed before the loader for C1 upsampling) ──
    sample_types = []
    if cfg.data.dataset_type == "impressions":
        print("Loading sample types for Impressions dataset")
        train_file = json.load(open(cfg.data.train_annotation_path))
        train_file = [train_file[i] for i in sample_ids_list]
        _type_map = {"caption": 0, "description": 1, "impression": 2, "aesthetic": 3}
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

    # ── Train dataset & loader ────────────────────────────────────────────────
    caption_upsample = getattr(cfg.train, "caption_upsample", 1.0)

    if feature_manager.fits_in_ram():
        print(
            f"RAM mode: loading {feature_manager.cls_features_size_gb():.1f} GiB of "
            "features into RAM for true-random batches."
        )
        train_set = CoSiRShardDataset(feature_manager, feature_types=feature_types)

        # C1: upsample caption via WeightedRandomSampler (map-style dataset only)
        if len(sample_types) > 0 and caption_upsample != 1.0:
            weights = np.ones(len(sample_types), dtype=np.float32)
            weights[sample_types == 0] = caption_upsample
            sampler = WeightedRandomSampler(
                weights.tolist(), len(weights), replacement=True
            )
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.train.batch_size,
                sampler=sampler,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
            )
            print(f"C1: caption upsampled {caption_upsample}x via WeightedRandomSampler")
        else:
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
        if caption_upsample != 1.0:
            print("Warning: caption_upsample ignored in stream mode (iterable dataset)")
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
        ratio=float(getattr(cfg.eval, "test_ratio", 1.0)),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    return train_set, train_loader, test_set, test_loader, sample_types


# ─── Evaluation snapshot helpers ────────────────────────────────────────────
# These are called from _eval_snapshot once per evaluation epoch.
# Each handles one distinct artifact type and can be extended independently.


def _save_condition_viz_snapshot(
    cfg,
    epoch,
    experiment,
    model,
    all_img_emb,
    all_txt_emb,
    all_raw_text,
    image_to_text_map,
    text_to_image_map,
    test_set,
    label_embeddings_all,
    representatives,
    sample_types,
):
    """Save per-epoch condition viz data for the interactive analysis notebooks."""
    print("Saving condition visualization snapshot...")
    cond_viz_dir = experiment.directory / "condition_viz"
    cond_viz_dir.mkdir(parents=True, exist_ok=True)

    # Fixed embeddings saved once — backbone is frozen so these never change
    fixed_path = cond_viz_dir / "fixed_data.pt"
    if not fixed_path.exists():
        image_paths = [
            os.path.join(test_set.image_path, test_set.annotations[i]["image"])
            for i in range(all_img_emb.shape[0])
        ]
        _test_caption_types = None
        if cfg.data.dataset_type == "impressions":
            _type_map = {"caption": 0, "description": 1, "impression": 2, "aesthetic": 3}
            _flat = []
            for _i in range(all_img_emb.shape[0]):
                for _t in test_set.annotations[_i]["caption_type"]:
                    _flat.append(_type_map[_t])
            _test_caption_types = torch.tensor(_flat, dtype=torch.long)
        torch.save(
            {
                "all_img_emb": all_img_emb.cpu(),
                "all_txt_emb": all_txt_emb.cpu(),
                "all_raw_text": all_raw_text,
                "image_paths": image_paths,
                "image_to_text_map": image_to_text_map.cpu(),
                "text_to_image_map": text_to_image_map.cpu(),
                "captions_per_image": test_set.captions_per_image,
                "test_caption_types": _test_caption_types,
            },
            fixed_path,
        )
        print(f"  Saved condition viz fixed data → {fixed_path}")

    # Per-epoch: conditions + model weights (replace on each call)
    epoch_path = cond_viz_dir / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "label_embeddings_all": label_embeddings_all.cpu(),
            "representatives": representatives.cpu(),
            "combiner_state_dict": model.combiner.state_dict(),
            "predictor_state_dict": model.condition_predictor.state_dict(),
            "other_proj_state_dict": model.other_proj.state_dict(),
            "combine_side": cfg.model.combine_side,
            "combiner_config": {
                "clip_feature_dim": model.feature_dim,
                "projection_dim": model.feature_dim,
                "label_dim": cfg.model.embedding_dim,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
            },
            "predictor_config": {
                "input_dim": model.feature_dim,
                "hidden_dim": cfg.model.hidden_dim,
                "output_dim": cfg.model.embedding_dim,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
            },
            "other_proj_config": {
                "feature_dim": model.feature_dim,
                "type": type(model.other_proj).__name__,
                "hidden_dim": getattr(model.other_proj, "hidden_dim", model.feature_dim),
                "num_blocks": getattr(model.other_proj, "num_blocks", 3),
            },
            "train_sample_types": (
                torch.tensor(sample_types, dtype=torch.long) if len(sample_types) > 0 else None
            ),
        },
        epoch_path,
    )
    print(f"  Saved condition viz epoch snapshot → {epoch_path}")


def _save_condition_analysis_cache(
    cfg,
    epoch,
    experiment,
    model,
    all_img_emb,
    all_txt_emb,
    text_to_image_map,
    image_to_text_map,
    representatives,
    device,
):
    """Compute and save per-representative GT ranks and top-K for the full test set.

    Used by the deep retrieval analysis notebooks.
    """
    print("Saving condition analysis cache...")
    ca_dir = experiment.directory / "condition_analysis"
    ca_dir.mkdir(parents=True, exist_ok=True)
    CA_TOPK = 10

    n_txt = all_txt_emb.shape[0]
    n_img = all_img_emb.shape[0]
    K = len(representatives)
    bs = max(cfg.train.batch_size, 256)

    txt_t = all_txt_emb.to(device)
    img_t = all_img_emb.to(device)
    img_n = F.normalize(img_t, dim=-1)
    txt_n = F.normalize(txt_t, dim=-1)
    # Projected "other side" for oracle sims; raw tensors kept for CLIP baseline
    if cfg.model.combine_side == "txt":
        other_n = F.normalize(model.project_other(img_t), dim=-1)
    else:
        other_n = F.normalize(model.project_other(txt_t), dim=-1)
    ttimap = text_to_image_map.cpu()   # [N_txt]
    ittmap = image_to_text_map.cpu()   # [N_img, cpi]

    per_rep_gt_rank = torch.zeros(K, n_txt, dtype=torch.long)
    per_rep_topk_idx = torch.zeros(K, n_txt, CA_TOPK, dtype=torch.long)
    per_rep_topk_scores = torch.zeros(K, n_txt, CA_TOPK)
    per_rep_i2t_gt_rank = torch.zeros(K, n_img, dtype=torch.long)
    per_rep_i2t_topk_idx = torch.zeros(K, n_img, CA_TOPK, dtype=torch.long)
    per_rep_i2t_topk_scores = torch.zeros(K, n_img, CA_TOPK)
    run_mean: Optional[torch.Tensor] = None
    run_max: Optional[torch.Tensor] = None
    i2t_run_mean: Optional[torch.Tensor] = None
    i2t_run_max: Optional[torch.Tensor] = None

    with torch.no_grad():
        for ri in range(K):
            cond = representatives[ri].unsqueeze(0).to(device)
            if cfg.model.combine_side == "txt":
                chunks = []
                for i in range(0, n_txt, bs):
                    e = min(i + bs, n_txt)
                    chunks.append(model.combine(txt_t[i:e], None, cond.expand(e - i, -1)))
                comb = F.normalize(torch.cat(chunks, dim=0), dim=-1)
                sims = (comb @ other_n.T).cpu()            # [N_txt, N_img]
            else:
                chunks = []
                for i in range(0, n_img, bs):
                    e = min(i + bs, n_img)
                    chunks.append(model.combine(img_t[i:e], None, cond.expand(e - i, -1)))
                comb = F.normalize(torch.cat(chunks, dim=0), dim=-1)
                sims = (other_n @ comb.T).cpu()            # [N_txt, N_img]
            del chunks, comb
            torch.cuda.empty_cache()

            # T2I: each text finds its GT image
            gt_s = sims[torch.arange(n_txt), ttimap]
            per_rep_gt_rank[ri] = (sims >= gt_s.unsqueeze(1)).sum(dim=1).long() - 1
            tk = torch.topk(sims, k=CA_TOPK, dim=1)
            per_rep_topk_idx[ri] = tk.indices
            per_rep_topk_scores[ri] = tk.values

            # I2T: each image finds its best GT text
            sims_i2t = sims.T                              # [N_img, N_txt]
            i2t_gt = sims_i2t[torch.arange(n_img).unsqueeze(1), ittmap]
            i2t_best = i2t_gt.max(dim=1).values
            per_rep_i2t_gt_rank[ri] = (sims_i2t >= i2t_best.unsqueeze(1)).sum(dim=1).long() - 1
            tk_i2t = torch.topk(sims_i2t, k=CA_TOPK, dim=1)
            per_rep_i2t_topk_idx[ri] = tk_i2t.indices
            per_rep_i2t_topk_scores[ri] = tk_i2t.values

            if run_mean is None:
                run_mean = sims.clone()
                run_max = sims.clone()
                i2t_run_mean = sims_i2t.clone()
                i2t_run_max = sims_i2t.clone()
            else:
                run_mean.add_(sims)
                torch.maximum(run_max, sims, out=run_max)  # type: ignore[arg-type]
                i2t_run_mean.add_(sims_i2t)               # type: ignore[union-attr]
                torch.maximum(i2t_run_max, sims_i2t, out=i2t_run_max)  # type: ignore[arg-type]
            del sims, sims_i2t, i2t_gt, i2t_best, tk_i2t

    run_mean.div_(K)        # type: ignore[union-attr]
    i2t_run_mean.div_(K)    # type: ignore[union-attr]
    oracle = per_rep_gt_rank.argmin(dim=0)           # [N_txt]
    i2t_oracle = per_rep_i2t_gt_rank.argmin(dim=0)  # [N_img]
    mean_tk = torch.topk(run_mean, k=CA_TOPK, dim=1)          # type: ignore[arg-type]
    max_tk = torch.topk(run_max, k=CA_TOPK, dim=1)            # type: ignore[arg-type]
    i2t_mean_tk = torch.topk(i2t_run_mean, k=CA_TOPK, dim=1)  # type: ignore[arg-type]
    i2t_max_tk = torch.topk(i2t_run_max, k=CA_TOPK, dim=1)    # type: ignore[arg-type]

    # CLIP baseline GT ranks (no condition) for both T2I and I2T
    clip_sims = (txt_n @ img_n.T).cpu()                        # [N_txt, N_img]
    clip_gt_s = clip_sims[torch.arange(n_txt), ttimap]
    clip_gt_rank = (clip_sims >= clip_gt_s.unsqueeze(1)).sum(dim=1).long() - 1
    clip_sims_i2t = clip_sims.T
    clip_i2t_gt = clip_sims_i2t[torch.arange(n_img).unsqueeze(1), ittmap]
    clip_i2t_best = clip_i2t_gt.max(dim=1).values
    clip_i2t_gt_rank = (clip_sims_i2t >= clip_i2t_best.unsqueeze(1)).sum(dim=1).long() - 1
    del clip_sims, clip_gt_s, clip_sims_i2t, clip_i2t_gt, clip_i2t_best

    torch.save(
        {
            "epoch": epoch,
            "n_representatives": K,
            # T2I
            "per_rep_gt_rank": per_rep_gt_rank,               # [K, N_txt]
            "per_rep_topk_indices": per_rep_topk_idx,         # [K, N_txt, top_k]
            "per_rep_topk_scores": per_rep_topk_scores,       # [K, N_txt, top_k]
            "oracle_condition_idx": oracle,                   # [N_txt]
            "mean_topk_indices": mean_tk.indices,             # [N_txt, top_k]
            "mean_topk_scores": mean_tk.values,
            "max_topk_indices": max_tk.indices,
            "max_topk_scores": max_tk.values,
            "clip_gt_rank": clip_gt_rank,                     # [N_txt]
            # I2T
            "per_rep_i2t_gt_rank": per_rep_i2t_gt_rank,      # [K, N_img]
            "per_rep_i2t_topk_indices": per_rep_i2t_topk_idx,
            "per_rep_i2t_topk_scores": per_rep_i2t_topk_scores,
            "i2t_oracle_condition_idx": i2t_oracle,
            "i2t_mean_topk_indices": i2t_mean_tk.indices,
            "i2t_mean_topk_scores": i2t_mean_tk.values,
            "i2t_max_topk_indices": i2t_max_tk.indices,
            "i2t_max_topk_scores": i2t_max_tk.values,
            "clip_i2t_gt_rank": clip_i2t_gt_rank,
        },
        ca_dir / f"epoch_{epoch:04d}.pt",
    )
    print(f"  Saved condition analysis → {ca_dir / f'epoch_{epoch:04d}.pt'}")

    # Free GPU tensors explicitly; this function is memory-intensive
    del txt_t, img_t, img_n, txt_n, other_n
    torch.cuda.empty_cache()


def _save_retrieval_snapshot(
    cfg,
    epoch,
    experiment,
    model,
    all_img_emb,
    all_txt_emb,
    all_raw_text,
    image_to_text_map,
    text_to_image_map,
    test_set,
    representatives,
    device,
):
    """Save qualitative retrieval top-K per query for cross-epoch comparison.

    Writes one .pt per epoch plus a one-time metadata.pt with the CLIP baseline.
    """
    print("Saving retrieval snapshot...")
    N_FIXED = 50   # first N image/text queries, fixed across all epochs
    TOP_K = 10     # top-K results stored per query

    snap_dir = experiment.directory / "retrieval_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    n_img = all_img_emb.shape[0]
    n_txt = all_txt_emb.shape[0]
    nfi = min(N_FIXED, n_img)
    nft = min(N_FIXED, n_txt)

    img_n = F.normalize(all_img_emb.to(device), dim=-1)   # [N_img, D]
    txt_raw = all_txt_emb.to(device)                      # [N_txt, D]
    txt_n = F.normalize(txt_raw, dim=-1)                  # [N_txt, D]
    img_raw = all_img_emb.to(device)                      # [N_img, D]
    fixed_img_n = img_n[:nfi]
    fixed_txt_n = txt_n[:nft]
    # Projected "other side" for model sims; raw img_n/txt_n kept for CLIP baseline
    if cfg.model.combine_side == "txt":
        other_n = F.normalize(model.project_other(img_raw), dim=-1)
        fixed_other_n = other_n[:nfi]
    else:
        other_n = F.normalize(model.project_other(txt_raw), dim=-1)
        fixed_other_n = other_n[:nft]

    run_max_i2t = None   # [nfi, N_txt]
    run_max_t2i = None   # [nft, N_img]

    for rep in representatives:
        cond = rep.unsqueeze(0).to(device)
        if cfg.model.combine_side == "txt":
            comb_mod = model.combine(txt_raw, None, cond.expand(n_txt, -1))
            sim_i2t = (fixed_other_n @ comb_mod.T).cpu()
            sim_t2i = (comb_mod[:nft] @ other_n.T).cpu()
        else:
            comb_mod = model.combine(img_raw, None, cond.expand(n_img, -1))
            sim_i2t = (comb_mod[:nfi] @ other_n.T).cpu()
            sim_t2i = (fixed_other_n @ comb_mod.T).cpu()
        run_max_i2t = sim_i2t if run_max_i2t is None else torch.maximum(run_max_i2t, sim_i2t)
        run_max_t2i = sim_t2i if run_max_t2i is None else torch.maximum(run_max_t2i, sim_t2i)

    ki2t = min(TOP_K, n_txt)
    kt2i = min(TOP_K, n_img)
    top_i2t = torch.topk(run_max_i2t, k=ki2t, dim=1).indices   # type: ignore[arg-type]
    top_t2i = torch.topk(run_max_t2i, k=kt2i, dim=1).indices   # type: ignore[arg-type]

    ittmap = image_to_text_map.cpu()   # [N_img, cpi]
    ttimap = text_to_image_map.cpu()   # [N_txt]

    is_gt_i2t = torch.zeros(nfi, ki2t, dtype=torch.bool)
    for q in range(nfi):
        gt_set = set(ittmap[q].tolist())
        for kp, tidx in enumerate(top_i2t[q].tolist()):
            if tidx in gt_set:
                is_gt_i2t[q, kp] = True

    is_gt_t2i = torch.zeros(nft, kt2i, dtype=torch.bool)
    for q in range(nft):
        gt_img = ttimap[q].item()
        for kp, iidx in enumerate(top_t2i[q].tolist()):
            if iidx == gt_img:
                is_gt_t2i[q, kp] = True

    epoch_path = snap_dir / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "combine_side": cfg.model.combine_side,
            "i2t": {"query_indices": list(range(nfi)), "top_k": top_i2t, "is_gt": is_gt_i2t},
            "t2i": {"query_indices": list(range(nft)), "top_k": top_t2i, "is_gt": is_gt_t2i},
        },
        epoch_path,
    )
    print(f"  Saved retrieval snapshot → {epoch_path}")

    # Metadata (captions, GT maps, image paths, CLIP baseline) — written once
    meta_path = snap_dir / "metadata.pt"
    if not meta_path.exists():
        image_paths = [
            os.path.join(test_set.image_path, test_set.annotations[i]["image"])
            for i in range(n_img)
        ]
        clip_sim_i2t = (fixed_img_n @ txt_n.T).cpu()   # [nfi, N_txt]
        clip_sim_t2i = (fixed_txt_n @ img_n.T).cpu()   # [nft, N_img]
        clip_top_i2t = torch.topk(clip_sim_i2t, k=ki2t, dim=1).indices
        clip_top_t2i = torch.topk(clip_sim_t2i, k=kt2i, dim=1).indices

        clip_is_gt_i2t = torch.zeros(nfi, ki2t, dtype=torch.bool)
        for q in range(nfi):
            gt_set = set(ittmap[q].tolist())
            for kp, tidx in enumerate(clip_top_i2t[q].tolist()):
                if tidx in gt_set:
                    clip_is_gt_i2t[q, kp] = True

        clip_is_gt_t2i = torch.zeros(nft, kt2i, dtype=torch.bool)
        for q in range(nft):
            gt_img = ttimap[q].item()
            for kp, iidx in enumerate(clip_top_t2i[q].tolist()):
                if iidx == gt_img:
                    clip_is_gt_t2i[q, kp] = True

        torch.save(
            {
                "captions": all_raw_text,
                "image_to_text_map": ittmap,
                "text_to_image_map": ttimap,
                "captions_per_image": test_set.captions_per_image,
                "n_images": n_img,
                "n_texts": n_txt,
                "image_paths": image_paths,
                "combine_side": cfg.model.combine_side,
                "clip_baseline": {
                    "i2t": {"top_k": clip_top_i2t, "is_gt": clip_is_gt_i2t},
                    "t2i": {"top_k": clip_top_t2i, "is_gt": clip_is_gt_t2i},
                },
            },
            meta_path,
        )
        print(f"  Saved retrieval metadata → {meta_path}")


def _eval_snapshot(
    cfg,
    epoch,
    model,
    embedding_manager,
    experiment,
    evaluator,
    umap_vis,
    sample_types,
    test_loader,
    test_set,
    logger,
    processor,
    device,
):
    """Run test evaluation, UMAP viz, auto-evaluator, and all per-epoch snapshots.

    Called at each evaluation epoch and at the final epoch.
    """
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        print("Getting all embeddings")
        _, label_embeddings_all = embedding_manager.get_all_embeddings()

        print("Getting representatives")
        # Use more representatives at the final epoch for richer analysis
        n_rep = (
            cfg.train.representative_number
            if epoch != cfg.train.epochs - 1
            else 30
        )
        representatives = get_representatives_fps(label_embeddings_all.cpu(), n_rep)
        print(f"Evaluating with {len(representatives)} representatives")

        test_detail = evaluator.evaluate_test(
            model=model,
            processor=processor,
            dataloader=test_loader,
            label_embeddings=representatives,
            epoch=epoch,
            return_detailed_results=True,
            use_oracle=True,
            oracle_aggregation=cfg.eval.oracle_aggregation,
        )
        test_detail = cast(TestEvaluationDetail, test_detail)
        all_img_emb = test_detail.all_img_emb
        all_txt_emb = test_detail.all_txt_emb
        all_raw_text = test_detail.all_raw_text
        text_to_image_map = test_detail.text_to_image_map
        image_to_text_map = test_detail.image_to_text_map

        logger.log_test(test_detail.results, epoch=epoch)

        # ── UMAP visualization ──
        if label_embeddings_all.shape[1] == 2:
            umap_features = label_embeddings_all.cpu().numpy()
        else:
            umap_features = umap_vis.learn_umap(label_embeddings_all, close_cluster=True)

        # Map representatives into UMAP space by nearest-neighbour lookup
        rep_indices = (
            torch.cdist(
                representatives.float().cpu(),
                label_embeddings_all.float().cpu(),
            )
            .argmin(dim=1)
            .numpy()
        )
        umap_representatives = umap_features[rep_indices]

        fig = get_umap(
            umap_features,
            umap_labels=None,
            epoch=epoch,
            no_outlier=True,
            samples_to_track=[0, 1, 2, 3, 4],
            representatives=umap_representatives,
        )
        experiment.save_artifact(
            name=f"label_embeddings_umap_{epoch}",
            data=fig,
            artifact_type="figure",
            folder="plots",
            description=f"UMAP visualization of trained label embeddings at epoch {epoch}",
        )

        fig_3 = None
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

        logger.log(
            {
                "vis/umap": wandb.Image(fig),
                "vis/ideal_condition_space": wandb.Image(fig2),
                "vis/ground_truth_sample_types": (
                    wandb.Image(fig_3) if fig_3 is not None else None
                ),
                "test_epoch": epoch,
            }
        )
        plt.close("all")

        # ── Condition viz snapshot ──
        _save_condition_viz_snapshot(
            cfg,
            epoch,
            experiment,
            model,
            all_img_emb,
            all_txt_emb,
            all_raw_text,
            image_to_text_map,
            text_to_image_map,
            test_set,
            label_embeddings_all,
            representatives,
            sample_types,
        )

        # ── Automatic evaluator ──
        print("Evaluating automatic evaluator")
        auto_eval = CoSiRAutomaticEvaluator(
            model,
            (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
            label_embeddings_all,
            device,
            representatives=representatives,
        )
        result = auto_eval.evaluate_all()
        del auto_eval
        torch.cuda.empty_cache()

        _eval_keys = {
            "magnitude_effect": ["correlation"],
            "condition_distance_correlation": ["spearman_rho"],
            "retrieval_gain": ["R@1_absolute_gain", "R@1_baseline", "R@1_conditional"],
            "diversity": ["mean_jsd"],
            "best_condition_upper_bound": ["R@1_boost", "R@1_best_condition"],
            "space_quality": ["silhouette_score", "n_effective_dims"],
        }
        eval_metrics = {
            f"{group}/{key}": result[group][key]
            for group, keys in _eval_keys.items()
            if result.get(group)
            for key in keys
            if key in result[group]
        }
        logger.log_eval(eval_metrics, epoch=epoch)

        # ── Phase 1 eval metrics: direction_sim + per-type T2I R@1 ──
        if cfg.data.dataset_type == "impressions" and len(sample_types) > 0:
            _TYPE_NAMES = ["cap", "des", "imp", "aes"]
            _type_map_test = {"caption": 0, "description": 1, "impression": 2, "aesthetic": 3}
            _test_types_list = []
            for _i in range(all_img_emb.shape[0]):
                for _tt in test_set.annotations[_i]["caption_type"]:
                    _test_types_list.append(_type_map_test[_tt])
            _test_types = np.array(_test_types_list)  # [N_txt]
            _ttimap = text_to_image_map.cpu()

            _img_t = all_img_emb.to(device)
            _txt_t = all_txt_emb.to(device)
            _lab_all = label_embeddings_all.to(device)
            _n_img = _img_t.shape[0]
            _phase1_eval: dict = {}

            with torch.no_grad():
                _other_n = F.normalize(model.project_other(_txt_t), dim=-1)  # [N_txt, D]

                # Direction similarity — compare per-type mean condition shifts on a sample of images
                if cfg.model.combine_side == "img":
                    _n_s = min(256, _n_img)
                    _img_s = _img_t[:_n_s]
                    _img_s_n = F.normalize(_img_s, dim=-1)
                    _type_dmeans_eval = []
                    _present_types_eval = []
                    for _t in range(4):
                        _mask_t = sample_types == _t
                        if not _mask_t.any():
                            continue
                        _cond_t = _lab_all[_mask_t].mean(0).unsqueeze(0).expand(_n_s, -1)
                        _comb_t = F.normalize(model.combine(_img_s, None, _cond_t), dim=-1)
                        _type_dmeans_eval.append((_comb_t - _img_s_n).mean(0))
                        _present_types_eval.append(_t)

                    if len(_type_dmeans_eval) >= 2:
                        _dm_n_eval = F.normalize(torch.stack(_type_dmeans_eval), dim=-1)
                        _dir_mat = (_dm_n_eval @ _dm_n_eval.T).cpu().numpy()
                        _nt = len(_type_dmeans_eval)
                        _off = [(i, j) for i in range(_nt) for j in range(i + 1, _nt)]
                        _phase1_eval["direction_sim_mean"] = float(
                            np.mean([_dir_mat[i, j] for i, j in _off])
                        )
                        for _ia, _ta in enumerate(_present_types_eval):
                            for _ib, _tb in enumerate(_present_types_eval):
                                if _ib > _ia:
                                    _phase1_eval[
                                        f"direction_sim_{_TYPE_NAMES[_ta]}_{_TYPE_NAMES[_tb]}"
                                    ] = float(_dir_mat[_ia, _ib])

                # Per-type T2I R@1: predicted condition vs avg_all condition
                _pred_cond_e = model.predict_condition(_img_t)  # [N_img, D_cond]
                _comb_pred = F.normalize(model.combine(_img_t, None, _pred_cond_e), dim=-1)
                _sims_pred = (_other_n @ _comb_pred.T).cpu()  # [N_txt, N_img]
                _gt_s = _sims_pred[torch.arange(len(_ttimap)), _ttimap]
                _ranks_pred = (_sims_pred >= _gt_s.unsqueeze(1)).sum(1).long() - 1

                _avg_cond = _lab_all.mean(0).unsqueeze(0).expand(_n_img, -1)
                _comb_avg = F.normalize(model.combine(_img_t, None, _avg_cond), dim=-1)
                _sims_avg = (_other_n @ _comb_avg.T).cpu()
                _gt_s_avg = _sims_avg[torch.arange(len(_ttimap)), _ttimap]
                _ranks_avg = (_sims_avg >= _gt_s_avg.unsqueeze(1)).sum(1).long() - 1

                for _t, _name in enumerate(_TYPE_NAMES):
                    _mask_tt = _test_types == _t
                    if _mask_tt.any():
                        _phase1_eval[f"r1_predicted_{_name}"] = float(
                            (_ranks_pred[_mask_tt] == 0).float().mean()
                        )
                        _phase1_eval[f"r1_avg_all_{_name}"] = float(
                            (_ranks_avg[_mask_tt] == 0).float().mean()
                        )

            _phase1_log = {f"eval_phase1/{k}": v for k, v in _phase1_eval.items()}
            _phase1_log["eval_epoch"] = epoch
            logger.log(_phase1_log)

            del _img_t, _txt_t, _lab_all, _other_n, _comb_pred, _sims_pred, _comb_avg, _sims_avg
            torch.cuda.empty_cache()

        # ── Phase 2 eval: condition transfer score ──
        # Measures whether similar predicted conditions transfer retrieval benefit across
        # queries — answers whether the retrieval task is naturally creating coherent
        # condition structure without GT type labels.
        # Spearman ρ > 0 and rising → structure emerging; flat → no useful structure.
        _lambda_var = getattr(cfg.loss, "lambda_var", 0.0)
        _lambda_cov = getattr(cfg.loss, "lambda_cov", 0.0)
        if (_lambda_var > 0 or _lambda_cov > 0) and cfg.model.combine_side == "img":
            from scipy.stats import spearmanr as _spearmanr
            _TM = min(50, all_img_emb.shape[0])
            _img_m = all_img_emb[:_TM].to(device)
            _txt_all = all_txt_emb.to(device)
            _other_m = F.normalize(model.project_other(_txt_all), dim=-1)  # [N_txt, D]
            _ittmap_m = image_to_text_map[:_TM].to(device)                 # [TM, cpi]
            _lab_m = label_embeddings_all.to(device)
            _avg_cond_m = _lab_m.mean(0).unsqueeze(0).expand(_TM, -1)

            with torch.no_grad():
                # Predicted conditions for the TM images
                _pred_conds_m = model.predict_condition(_img_m)             # [TM, D_cond]
                _pred_conds_n = F.normalize(_pred_conds_m, dim=-1)
                _cond_sim_mat = (_pred_conds_n @ _pred_conds_n.T).cpu()    # [TM, TM]

                # Avg-all baseline rank for each image (I2T direction)
                _comb_avg_m = F.normalize(model.combine(_img_m, None, _avg_cond_m), dim=-1)
                _sims_avg_m = (_comb_avg_m @ _other_m.T)                   # [TM, N_txt]
                _gt_avg = _sims_avg_m[
                    torch.arange(_TM, device=device).unsqueeze(1), _ittmap_m
                ].max(dim=1).values                                         # [TM]
                _rank_avg_m = (_sims_avg_m >= _gt_avg.unsqueeze(1)).sum(1).long() - 1  # [TM]

                # Per-condition ranks: for each condition j, rank all TM images
                _rank_matrix = torch.zeros(_TM, _TM, dtype=torch.long)
                for _j in range(_TM):
                    _cond_j = _pred_conds_m[_j].unsqueeze(0).expand(_TM, -1)
                    _comb_j = F.normalize(model.combine(_img_m, None, _cond_j), dim=-1)
                    _sims_j = (_comb_j @ _other_m.T)                       # [TM, N_txt]
                    _gt_j = _sims_j[
                        torch.arange(_TM, device=device).unsqueeze(1), _ittmap_m
                    ].max(dim=1).values
                    _rank_matrix[:, _j] = (
                        (_sims_j >= _gt_j.unsqueeze(1)).sum(1).long() - 1
                    ).cpu()

                # transfer_gain[i, j] = improvement for image i using condition j vs avg_all
                _transfer_gain = _rank_avg_m.cpu().unsqueeze(1) - _rank_matrix   # [TM, TM], positive = better

                # Spearman ρ over upper triangle (exclude self-pairs on diagonal)
                _triu_idx = torch.triu_indices(_TM, _TM, offset=1)
                _sim_flat = _cond_sim_mat[_triu_idx[0], _triu_idx[1]].numpy()
                _gain_flat = _transfer_gain[_triu_idx[0], _triu_idx[1]].float().numpy()
                if _sim_flat.std() > 1e-6 and _gain_flat.std() > 1e-6:
                    _rho, _ = _spearmanr(_sim_flat, _gain_flat)
                else:
                    _rho = 0.0
                logger.log({"eval_phase2/condition_transfer_rho": float(_rho), "eval_epoch": epoch})

            del _img_m, _txt_all, _other_m, _lab_m, _pred_conds_m, _cond_sim_mat
            del _comb_avg_m, _sims_avg_m, _rank_matrix, _transfer_gain
            torch.cuda.empty_cache()

        # ── Condition analysis cache ──
        _save_condition_analysis_cache(
            cfg,
            epoch,
            experiment,
            model,
            all_img_emb,
            all_txt_emb,
            text_to_image_map,
            image_to_text_map,
            representatives,
            device,
        )

        # ── Retrieval snapshot ──
        _save_retrieval_snapshot(
            cfg,
            epoch,
            experiment,
            model,
            all_img_emb,
            all_txt_emb,
            all_raw_text,
            image_to_text_map,
            text_to_image_map,
            test_set,
            representatives,
            device,
        )


def _save_final_artifacts(model, embedding_manager, experiment, cfg):
    """Copy final embeddings and save model checkpoints after training completes."""
    # Copy memmap files so phase-2 scripts can load without re-running training
    embedding_manager._copy_to(pathlib.Path(experiment.directory) / "final_embeddings")

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
        data={
            "combiner_state_dict": model.combiner.state_dict(),
            "predictor_state_dict": model.condition_predictor.state_dict(),
            "other_proj_state_dict": model.other_proj.state_dict(),
            "combine_side": cfg.model.combine_side,
            "combiner_config": {
                "clip_feature_dim": model.feature_dim,
                "projection_dim": model.feature_dim,
                "label_dim": cfg.model.embedding_dim,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
            },
            "predictor_config": {
                "input_dim": model.feature_dim,
                "hidden_dim": cfg.model.hidden_dim,
                "output_dim": cfg.model.embedding_dim,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
            },
            "other_proj_config": {
                "feature_dim": model.feature_dim,
                "type": type(model.other_proj).__name__,
                "hidden_dim": getattr(model.other_proj, "hidden_dim", model.feature_dim),
                "num_blocks": getattr(model.other_proj, "num_blocks", 3),
            },
        },
        artifact_type="torch",
        description="Phase 1 model: combiner + condition predictor state dictionaries",
    )


# ─── Entry point ────────────────────────────────────────────────────────────


def train_cosir(cfg, logger):
    # Reproducibility
    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # --- Phase 1: Model & Criteria ---
    evaluation_config, model, processor, criteria = _setup_model_and_criteria(cfg, device)

    # --- Phase 2: Features ---
    feature_manager, sample_ids_list = _extract_or_load_features(cfg, model, processor, device)

    # --- Phase 3: Experiment & Evaluators ---
    experiment, evaluator, umap_vis = _setup_experiment(cfg, evaluation_config, device, logger)

    # --- Phase 4: Embeddings ---
    # Initialized before the optimizer so the parameter reference is stable after
    # template loading (which may replace embedding_manager.embeddings).
    embedding_manager = _init_embedding_manager(
        cfg, device, sample_ids_list, experiment, feature_manager, model
    )

    # --- Phase 5: Optimizer & Scheduler ---
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, model, embedding_manager)

    # Families #1 & #2 share the persisted E edge list + CSR neighbour index.
    # Disabled (no-op) when both lambdas are 0 or edges are absent.
    _lambda_buddy = getattr(cfg.loss, "lambda_buddy", 0.0)
    _buddy_reg_samples = int(getattr(cfg.loss, "buddy_reg_samples", 4))
    _lambda_buddy_con = getattr(cfg.loss, "lambda_buddy_con", 0.0)
    _buddy_con_samples = int(getattr(cfg.loss, "buddy_con_samples", 4))
    _buddy_con_temp = float(getattr(cfg.loss, "buddy_con_temperature", 0.07))
    _log_buddy_preservation = bool(getattr(cfg.loss, "log_buddy_preservation", False))
    _buddy_preservation_k = int(getattr(cfg.loss, "buddy_preservation_k", 10))
    buddy_indptr = buddy_indices = None
    _clip_indptr = _clip_indices = None   # stable CLIP CSR (never rebound by refresh)
    other_feat_table = None
    _clip_edge_index = None
    if _lambda_buddy > 0 or _lambda_buddy_con > 0 or _log_buddy_preservation:
        _edges = embedding_manager.get_buddy_edges()
        if _edges is None:
            print("[buddy] lambda_buddy/lambda_buddy_con>0 but no buddy_edges.npy "
                  "found — disabling buddy terms for this run.")
            _lambda_buddy = 0.0
            _lambda_buddy_con = 0.0
        else:
            _edge_index = torch.from_numpy(_edges.astype(np.int64)).to(device)
            _clip_edge_index = _edge_index
            buddy_indptr, buddy_indices = build_neighbor_csr(
                _edge_index, num_nodes=len(embedding_manager.sample_ids)
            )
            _clip_indptr, _clip_indices = buddy_indptr, buddy_indices
            print(f"[buddy] edges loaded: {_edge_index.shape[1]:,}; "
                  f"lambda_buddy={_lambda_buddy}, lambda_buddy_con={_lambda_buddy_con}")

    # Family #2: gather the non-combine-side pooled feature per sample, in z-table
    # order, so anchor combined features can be pulled toward buddy targets.
    if _lambda_buddy_con > 0 and buddy_indptr is not None:
        if not feature_manager.fits_in_ram():
            print("[buddy-con] feature store does not fit in RAM (streaming path) — "
                  "buddy contrastive term unsupported here; disabling.")
            _lambda_buddy_con = 0.0
        else:
            _other_key = "txt_features" if cfg.model.combine_side == "img" else "img_features"
            _feat = feature_manager.load_all_to_ram([_other_key])
            other_feat_table = reorder_features_to_z(
                _feat[_other_key],
                [int(s) for s in _feat["sample_ids"].tolist()],
                embedding_manager.sample_ids,
            ).to(device)
            print(f"[buddy-con] enabled: lambda_buddy_con={_lambda_buddy_con}, "
                  f"samples/anchor={_buddy_con_samples}, temp={_buddy_con_temp}, "
                  f"other_feat={_other_key} {tuple(other_feat_table.shape)}")

    # Family #3: self-refreshing buddy graph. Recompute the CSR fed to the #2
    # contrastive term from the evolving combined features on a schedule.
    _buddy_refresh = bool(getattr(cfg.loss, "buddy_refresh", False))
    _buddy_refresh_warmup = int(getattr(cfg.loss, "buddy_refresh_warmup", 50))
    _buddy_refresh_period = int(getattr(cfg.loss, "buddy_refresh_period", 50))
    _buddy_refresh_blend = float(getattr(cfg.loss, "buddy_refresh_blend", 1.0))
    _buddy_refresh_k = int(getattr(cfg.loss, "buddy_refresh_k", 30))
    combine_feat_table = None
    _refresh_gen = torch.Generator().manual_seed(0)
    _prev_comb_edges = None

    # The combine-side pooled feature table (z-order) is needed by BOTH the refresh
    # (#3) and the buddy-preservation diagnostic; build it once if either wants it.
    if _buddy_refresh or _log_buddy_preservation:
        if _clip_edge_index is None:
            print("[buddy] refresh / buddy-preservation requested but no buddy edges — "
                  "disabling both.")
            _buddy_refresh = False
            _log_buddy_preservation = False
        elif not feature_manager.fits_in_ram():
            print("[buddy] combine-side feature table needs a RAM feature store "
                  "(streaming path) — disabling refresh and buddy-preservation.")
            _buddy_refresh = False
            _log_buddy_preservation = False
        else:
            _combine_key = "img_features" if cfg.model.combine_side == "img" else "txt_features"
            _cfeat = feature_manager.load_all_to_ram([_combine_key])
            combine_feat_table = reorder_features_to_z(
                _cfeat[_combine_key],
                [int(s) for s in _cfeat["sample_ids"].tolist()],
                embedding_manager.sample_ids,
            ).to(device)

    # Family #3 refresh needs a live #2 term (its frozen target table); guard on it.
    if _buddy_refresh:
        if _lambda_buddy_con <= 0 or other_feat_table is None:
            print("[buddy-refresh] requires lambda_buddy_con>0 with buddy edges — "
                  "disabling refresh for this run.")
            _buddy_refresh = False
        else:
            print(f"[buddy-refresh] enabled: warmup={_buddy_refresh_warmup}, "
                  f"period={_buddy_refresh_period}, blend={_buddy_refresh_blend}, "
                  f"k={_buddy_refresh_k}, combine_feat={_combine_key} "
                  f"{tuple(combine_feat_table.shape)}")
            if cfg.train.em_interval > 0:
                print("[buddy-refresh] WARNING: em_interval>0 (EM alternation) with "
                      "buddy_refresh: scheduled refreshes landing in a network phase are "
                      "skipped, not retried — refresh schedule may desync. Recommended: "
                      "run refresh with em_interval<0.")
            if _lambda_buddy > 0:
                print("[buddy-refresh] WARNING: lambda_buddy>0 with buddy_refresh: Family #1 "
                      "shares the refreshed CSR (out of scope). Recommended: lambda_buddy=0.")

    if _log_buddy_preservation and combine_feat_table is not None and _clip_indptr is not None:
        print(f"[buddy-preserve] enabled: buddy_knn_preservation@k={_buddy_preservation_k}, "
              f"logged at eval cadence (uses the frozen CLIP E graph).")
    else:
        _log_buddy_preservation = False

    # Snapshot the initial label embeddings so we can log mean drift ‖z − z_init‖
    # on the eval cadence. Logged for every run (incl. lambda_buddy=0) so the
    # baseline drift is the control: it shows how far training moves z away from
    # the init, and whether the buddy regularizer (lambda_buddy>0) reins that in.
    _z_init = embedding_manager.embeddings.detach().clone()

    # --- Phase 6: Data Loaders ---
    train_set, train_loader, test_set, test_loader, sample_types = _build_dataloaders(
        cfg, feature_manager, processor, sample_ids_list
    )

    # --- Training Loop ---
    global_step = 0
    em_interval = cfg.train.em_interval
    em_enabled = em_interval > 0
    _prev_em_phase: str | None = None  # force transition on epoch 0

    for epoch in range(cfg.train.epochs):
        experiment.current_epoch = epoch
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        cos_sim = None  # updated every 100 batches for cheap monitoring

        # EM phase alternation (disabled when em_interval < 0: both always update)
        if em_enabled:
            em_phase = "network" if (epoch // em_interval) % 2 == 0 else "conditions"
            if em_phase != _prev_em_phase:
                if em_phase == "network":
                    print(f"[EM] Epoch {epoch}: switching to NETWORK update phase")
                    for param in model.parameters():
                        param.requires_grad = True
                    embedding_manager.embeddings.requires_grad_(False)
                else:
                    print(f"[EM] Epoch {epoch}: switching to CONDITIONS update phase")
                    for param in model.parameters():
                        param.requires_grad = False
                    embedding_manager.embeddings.requires_grad_(True)
                _prev_em_phase = em_phase
        else:
            em_phase = "both"

        # Family #3: refresh the buddy graph feeding the #2 term, on schedule.
        if (
            _buddy_refresh
            and _lambda_buddy_con > 0
            and combine_feat_table is not None
            and embedding_manager.embeddings.requires_grad
            and epoch >= _buddy_refresh_warmup
            and (epoch - _buddy_refresh_warmup) % _buddy_refresh_period == 0
        ):
            buddy_indptr, buddy_indices, _comb_edges, _refresh_stats = refresh_buddy_graph(
                model,
                combine_feat_table,
                embedding_manager.embeddings,
                _clip_edge_index,
                num_nodes=len(embedding_manager.sample_ids),
                k=_buddy_refresh_k,
                blend=_buddy_refresh_blend,
                generator=_refresh_gen,
            )
            if _prev_comb_edges is not None:
                _refresh_stats["graph_churn"] = edge_jaccard(_comb_edges, _prev_comb_edges)
            _prev_comb_edges = _comb_edges
            logger.log_train(_refresh_stats, epoch=epoch, section="buddy_refresh")
            print(f"[buddy-refresh] epoch {epoch}: {_refresh_stats}")

        epoch_start_time = time.time()
        if isinstance(train_set, CoSiRShardStreamDataset):
            train_set.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            img_features = batch["img_features"].to(device, non_blocking=True)
            txt_features = batch["txt_features"].to(device, non_blocking=True)
            txt_full = (
                batch["txt_full"].to(device, non_blocking=True)
                if "txt_full" in batch
                else torch.zeros_like(txt_features)
            )
            img_full = (
                batch["img_full"].to(device, non_blocking=True)
                if "img_full" in batch
                else torch.zeros_like(img_features)
            )
            batch_sample_ids = batch["sample_ids"].tolist()

            # Differentiable slice — gradients flow back to embedding_manager.embeddings
            batch_indices = [embedding_manager.id_to_index[sid] for sid in batch_sample_ids]
            label_embeddings_before = embedding_manager.embeddings.data[batch_indices].clone()
            label_embeddings = embedding_manager.embeddings[batch_indices]

            if cfg.model.combine_side == "txt":
                combine_emb, combine_full = txt_features, txt_full
                loss_img_target, loss_txt_ref = img_features, txt_features
            else:
                combine_emb, combine_full = img_features, img_full
                loss_img_target, loss_txt_ref = txt_features, img_features

            other_emb = model.project_other(loss_img_target)

            # Oracle-guided advantage weighting: after backward, scale each condition's
            # gradient by softmax(advantage / tau) where advantage = sim_own - sim_rand.
            # Softmax keeps all conditions in play (no abandonment) while amplifying
            # conditions that are already more specialised than a random peer.
            _oracle_guided = getattr(cfg.train, "oracle_guided", False)
            _oracle_weights = None
            _oracle_frac = None
            _oracle_mean_adv = None
            if _oracle_guided:
                _tau = getattr(cfg.train, "oracle_advantage_tau", 0.1)
                with torch.no_grad():
                    _c_rand = label_embeddings[torch.randperm(len(label_embeddings))]
                    _other_n_probe = F.normalize(other_emb, dim=-1)
                    _sim_own = (
                        F.normalize(model.combine(combine_emb, None, label_embeddings, epoch=epoch), dim=-1)
                        * _other_n_probe
                    ).sum(-1)
                    _sim_rand = (
                        F.normalize(model.combine(combine_emb, None, _c_rand, epoch=epoch), dim=-1)
                        * _other_n_probe
                    ).sum(-1)
                    _advantage = _sim_own - _sim_rand  # [B], no clamp — all conditions stay active
                    _oracle_weights = F.softmax(_advantage / _tau, dim=0) * len(_advantage)  # mean=1
                    _oracle_frac = (_advantage > 0).float().mean().item()
                    _oracle_mean_adv = _advantage.mean().item()

            comb_emb, delta, gate_scalar, gate_logit = model.combine(
                combine_emb,
                combine_full,
                label_embeddings,
                epoch=epoch,
                return_label_proj=False,
                return_delta=True,
                return_scalar=True,
            )

            loss_dict = criteria(
                other_emb,
                loss_txt_ref,
                comb_emb,
                None,
                label_embeddings,
                model,
                delta=delta,
                scalar=gate_scalar,
                gate_logit=gate_logit,
            )

            if batch_idx % 100 == 0:
                cos_sim = torch.nn.functional.cosine_similarity(
                    comb_emb,
                    torch.nn.functional.normalize(combine_emb, dim=-1),
                    dim=-1,
                )

            loss = loss_dict["total_loss"]

            # Condition predictor distillation + L5 entropy diversity.
            # pred_cond is shared between both losses to avoid a second forward pass.
            lambda_pred = cfg.loss.lambda_pred
            lambda_ent = getattr(cfg.loss, "lambda_ent", 0.0)
            ent_tau = getattr(cfg.loss, "ent_tau", 5.0)

            pred_cond = None
            if lambda_pred > 0 or (lambda_ent > 0 and len(sample_types) > 0):
                pred_cond = model.predict_condition(combine_emb)

            if lambda_pred > 0 and pred_cond is not None:
                pred_loss = (
                    1
                    - F.cosine_similarity(pred_cond, label_embeddings.detach(), dim=-1)
                ).mean()
                loss = loss + lambda_pred * pred_loss
                loss_dict["loss_pred"] = pred_loss

            if lambda_ent > 0 and pred_cond is not None and len(sample_types) > 0:
                # L5: per-batch type-affinity distribution should be uniform across 4 types
                batch_types_arr = sample_types[np.array(batch_indices)]
                type_means_list = []
                for _t in range(4):
                    _mask = batch_types_arr == _t
                    if _mask.any():
                        type_means_list.append(label_embeddings.detach()[_mask].mean(0))
                    else:
                        type_means_list.append(label_embeddings.detach().mean(0))
                type_means_n = F.normalize(torch.stack(type_means_list), dim=-1)  # [4, D_cond]
                pred_n_l5 = F.normalize(pred_cond, dim=-1)  # [B, D_cond]
                batch_probs = F.softmax(pred_n_l5 @ type_means_n.T * ent_tau, dim=-1).mean(0)  # [4]
                pred_entropy = -(batch_probs * torch.log(batch_probs + 1e-8)).sum()
                loss = loss + lambda_ent * (-pred_entropy)  # maximise entropy
                loss_dict["pred_entropy"] = pred_entropy.detach()

            # VICReg diversity: variance + covariance on normalised conditions.
            # Applied on unit-sphere projections to be scale-invariant regardless of
            # whether label_embeddings are normalised during training.
            lambda_var = getattr(cfg.loss, "lambda_var", 0.0)
            lambda_cov = getattr(cfg.loss, "lambda_cov", 0.0)
            if lambda_var > 0 or lambda_cov > 0:
                _cond_n = F.normalize(label_embeddings, dim=-1)  # [B, D]
                _D = _cond_n.shape[1]
                if lambda_var > 0:
                    _var_gamma = getattr(cfg.loss, "var_gamma", 0.25)
                    _std = _cond_n.std(dim=0)  # [D]
                    var_loss = F.relu(_var_gamma - _std).mean()
                    loss = loss + lambda_var * var_loss
                    loss_dict["var_loss"] = var_loss.detach()
                if lambda_cov > 0:
                    _cond_c = _cond_n - _cond_n.mean(dim=0)
                    _cov = (_cond_c.T @ _cond_c) / (max(_cond_n.shape[0] - 1, 1))  # [D, D]
                    _off_diag = torch.ones(_D, _D, dtype=torch.bool, device=_cov.device).fill_diagonal_(False)
                    cov_loss = _cov[_off_diag].pow(2).mean()
                    loss = loss + lambda_cov * cov_loss
                    loss_dict["cov_loss"] = cov_loss.detach()

            # Gap alignment: condition pairwise similarities should match
            # (txt - img) pairwise similarities — a GT-free proxy for query style.
            # Same style → similar gap → similar conditions; works on any dataset.
            lambda_gap_align = getattr(cfg.loss, "lambda_gap_align", 0.0)
            if lambda_gap_align > 0:
                _gap = F.normalize(loss_img_target.detach() - combine_emb.detach(), dim=-1)  # [B, 512]
                _gap_sim = _gap @ _gap.T                                                      # [B, B]
                _cond_n_gap = F.normalize(label_embeddings, dim=-1)
                _cond_sim = _cond_n_gap @ _cond_n_gap.T                                      # [B, B]
                _off = ~torch.eye(len(_gap_sim), dtype=torch.bool, device=_gap_sim.device)
                gap_align_loss = F.mse_loss(_cond_sim[_off], _gap_sim[_off].detach())
                loss = loss + lambda_gap_align * gap_align_loss
                loss_dict["gap_align_loss"] = gap_align_loss.detach()

            # Family #1: buddy-graph smoothness on z along E (only when conditions train)
            if (
                _lambda_buddy > 0
                and buddy_indptr is not None
                and embedding_manager.embeddings.requires_grad
            ):
                _anchor_pos = torch.tensor(batch_indices, device=device, dtype=torch.long)
                buddy_loss = buddy_graph_smoothness_loss(
                    embedding_manager.embeddings,
                    buddy_indptr,
                    buddy_indices,
                    _anchor_pos,
                    num_samples=_buddy_reg_samples,
                )
                loss = loss + _lambda_buddy * buddy_loss
                loss_dict["loss_buddy"] = buddy_loss.detach()

            # Family #2: buddy contrastive supervision in combined/retrieval space
            if (
                _lambda_buddy_con > 0
                and other_feat_table is not None
                and embedding_manager.embeddings.requires_grad
            ):
                _anchor_pos_con = torch.tensor(batch_indices, device=device, dtype=torch.long)
                buddy_con_loss, buddy_con_align = buddy_contrastive_loss(
                    comb_emb,
                    _anchor_pos_con,
                    other_feat_table,
                    model.project_other,
                    other_emb,
                    buddy_indptr,
                    buddy_indices,
                    num_pos=_buddy_con_samples,
                    temperature=_buddy_con_temp,
                )
                loss = loss + _lambda_buddy_con * buddy_con_loss
                loss_dict["loss_buddy_con"] = buddy_con_loss.detach()
                loss_dict["buddy_con_alignment"] = buddy_con_align

            epoch_loss += loss.item()
            num_batches += 1

            # Phase-specific keys are routed to their own wandb sections
            _phase1_loss_keys = {"pred_entropy"}
            _phase2_loss_keys = {"var_loss", "cov_loss", "gap_align_loss"}
            _monitor_keys = {"diag_sim_gap", "off_diag_sim_gap", "total_sim_gap"}
            _phase_keys = _phase1_loss_keys | _phase2_loss_keys
            loss_metrics = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
                if k not in _monitor_keys and k not in _phase_keys
            }
            monitor_metrics = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
                if k in _monitor_keys
            }
            monitor_metrics["cos_sim"] = cos_sim.mean().item() if cos_sim is not None else None
            if _oracle_frac is not None:
                monitor_metrics["oracle_frac"] = _oracle_frac
                monitor_metrics["oracle_mean_adv"] = _oracle_mean_adv

            phase1_loss_metrics = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
                if k in _phase1_loss_keys
            }
            phase2_loss_metrics = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
                if k in _phase2_loss_keys
            }

            logger.log_train(loss_metrics, epoch=epoch, step=global_step, section="loss")
            logger.log_train(monitor_metrics, epoch=epoch, step=global_step, section="monitor")
            if phase1_loss_metrics:
                logger.log_train(phase1_loss_metrics, epoch=epoch, step=global_step, section="phase1_loss")
            if phase2_loss_metrics:
                logger.log_train(phase2_loss_metrics, epoch=epoch, step=global_step, section="phase2_loss")

            # Phase 1 monitor: batch-level direction diversity across condition types (every 50 steps)
            if batch_idx % 50 == 0 and len(sample_types) > 0:
                with torch.no_grad():
                    _btypes = sample_types[np.array(batch_indices)]
                    _comb_n = F.normalize(comb_emb.detach(), dim=-1)
                    _ref_n = F.normalize(combine_emb.detach(), dim=-1)
                    _deltas = _comb_n - _ref_n  # [B, D]
                    _type_dmeans = []
                    for _t in range(4):
                        _mask = _btypes == _t
                        if _mask.any():
                            _type_dmeans.append(_deltas[_mask].mean(0))
                    if len(_type_dmeans) >= 2:
                        _dm_n = F.normalize(torch.stack(_type_dmeans), dim=-1)
                        _n_t = _dm_n.shape[0]
                        _triu = torch.triu(
                            torch.ones(_n_t, _n_t, dtype=torch.bool, device=_dm_n.device),
                            diagonal=1,
                        )
                        _dir_sim = (_dm_n @ _dm_n.T)[_triu].mean().item()
                        logger.log_train(
                            {"direction_sim": _dir_sim},
                            epoch=epoch,
                            step=global_step,
                            section="phase1_monitor",
                        )
            logger.log_train(
                {"batch": batch_idx, "step": global_step},
                epoch=epoch,
                step=global_step,
                section="details",
            )

            optimizer.zero_grad()
            loss.backward()
            if _oracle_weights is not None and embedding_manager.embeddings.grad is not None:
                embedding_manager.embeddings.grad[batch_indices] *= _oracle_weights.unsqueeze(1)
            optimizer.step()
            global_step += 1

            if em_phase in ("conditions", "both"):
                if cfg.train.normalize:
                    with torch.no_grad():
                        embedding_manager.embeddings.data[batch_indices] = (
                            torch.nn.functional.normalize(
                                embedding_manager.embeddings.data[batch_indices], dim=-1
                            )
                        )
                label_embeddings_diff = (
                    (
                        embedding_manager.embeddings.data[batch_indices].cpu()
                        - label_embeddings_before.cpu()
                    )
                    .norm(dim=-1)
                    .mean()
                )
                logger.log_train(
                    {"label_embeddings_diff": label_embeddings_diff.item()},
                    epoch=epoch,
                    step=global_step,
                    section="monitor",
                )

        # ── Epoch end ──
        epoch_time = time.time() - epoch_start_time
        embedding_manager._save_all_chunks_to_disk()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        logger.log_train(
            {"loss": avg_loss, "epoch_time": epoch_time, "em_phase": em_phase},
            epoch=epoch,
        )

        scheduler.step()

        # ── Evaluation snapshot (periodic + final epoch) ──
        eval_due = (
            cfg.train.epochs == 0  # test-only mode
            or epoch % cfg.eval.evaluation_interval == 0
            or epoch == cfg.train.epochs - 1
        )
        if eval_due:
            with torch.no_grad():
                _drift = (
                    (embedding_manager.embeddings.detach() - _z_init)
                    .norm(dim=1)
                    .mean()
                    .item()
                )
            logger.log_train(
                {"drift_from_init": _drift}, epoch=epoch, section="buddy_diag"
            )
            if _log_buddy_preservation:
                _comb_all = compute_comb_all_eval(
                    model, combine_feat_table, embedding_manager.embeddings
                )
                _pres = buddy_knn_preservation(
                    _comb_all, _clip_indptr, _clip_indices, k=_buddy_preservation_k
                )
                logger.log_train(
                    {"buddy_knn_preservation": _pres}, epoch=epoch, section="buddy_diag"
                )

        if cfg.eval.perform_evaluation and eval_due:
            _eval_snapshot(
                cfg,
                epoch,
                model,
                embedding_manager,
                experiment,
                evaluator,
                umap_vis,
                sample_types,
                test_loader,
                test_set,
                logger,
                processor,
                device,
            )

    # --- Final Artifacts ---
    _save_final_artifacts(model, embedding_manager, experiment, cfg)
    print("Training Complete!")
    return 0
