

def annealing_rewarmup_schedule(
    epoch: int,
    T_max: int = 10,
    initial_lr: float = 1e-3,
    lr_min: float = 1e-9,
):
    # Adjusted epoch count after warmup
    t = epoch / T_max
    cosine_decay = 0.5 * (1 + math.cos(math.pi * t / T_max))
    min_scale = lr_min / initial_lr
    return min_scale + (1 - min_scale) * cosine_decay


def train(cfg: DictConfig, **kwargs):
    model = kwargs["model"]
    train_dataloader = kwargs["train_dataloader"]
    criteria = kwargs["criteria"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    scheduler = kwargs["scheduler"]
    embedding_manager = kwargs["embedding_manager"]
    log_interval = cfg.train.log_interval
    wandb_run = kwargs["wandb_run"]

    model.train()
    epoch_metrics = {"loss": 0.0, "other_metrics": {}}

    for batch_id, batch in enumerate(tqdm(train_dataloader)):
        img_emb, txt_emb, txt_full, label_embedding, sample_id = batch
        img_emb, txt_emb, txt_full, label_embedding = (
            img_emb.squeeze(0),
            txt_emb.squeeze(0),
            txt_full.squeeze(0),
            label_embedding.squeeze(0),
        )

        img_emb, txt_emb, txt_full = (
            img_emb.to(device, non_blocking=True),
            txt_emb.to(device, non_blocking=True),
            txt_full.to(device, non_blocking=True),
        )

        label_embedding = (
            label_embedding.to(device, non_blocking=True)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        label_embedding_cp = label_embedding.clone().detach()

        base_lr = optimizer.param_groups[0]["lr"]
        optimizer.add_param_group(
            {"params": [label_embedding], "lr": base_lr * 5e4}
        )  # Add label embedding to optimizer

        comb_emb, label_embedding_proj = model.module.combine(
            txt_emb, txt_full, label_embedding, epoch=epoch, return_label_proj=True
        )
        label_embedding_neg = replace_with_most_different(label_embedding)

        comb_emb_neg = model.module.combine(
            txt_emb, txt_full, label_embedding_neg, epoch=epoch, return_label_proj=False
        )

        loss_dict = criteria(
            img_emb,
            txt_emb,
            comb_emb,
            comb_emb_neg,
            label_embedding,
            label_embedding_proj,
        )
        if epoch < WARM_UP_EPOCH:
            loss = loss_dict["early_loss"]
        else:
            loss = loss_dict["total_loss"]

        epoch_metrics["loss"] += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer.param_groups.pop()  # Popout label embedding from optimizer

        # if update_label_embedding:
        embedding_manager.update_chunk_embeddings(batch_id, sample_id, label_embedding)

        # Log
        if batch_id % log_interval == 0 or batch_id == len(train_dataloader) - 1:
            label_move_distance = torch.norm(
                label_embedding - label_embedding_cp, p=2, dim=1
            ).mean()
            emb1_norm = F.normalize(comb_emb, p=2, dim=1)
            emb2_norm = F.normalize(comb_emb_neg, p=2, dim=1)
            sim_pos_neg = emb1_norm @ emb2_norm.T
            triu_indices = torch.triu_indices(
                sim_pos_neg.size(0), sim_pos_neg.size(1), offset=1
            )
            upper_vals = sim_pos_neg[triu_indices[0], triu_indices[1]]
            pos_neg_diff = torch.mean(upper_vals).item()
            print(
                f"Epoch: {epoch}, Batch: {batch_id} / {len(train_dataloader)-1 }, Loss: {loss.item():.3f}, Dynamic Scalar: {model.module.combiner.print_scalar()}, Cosine Similarity (pos-neg): {pos_neg_diff:.4f}, Label Move Distance: {label_move_distance:.7f}"
            )

        wandb_run.log(
            {
                "train/epoch": epoch,
                "train/total_loss": loss_dict["total_loss"].item(),
                "train/loss_improve": loss_dict["loss_improve"].item(),
                "train/loss_neg": loss_dict["loss_neg"].item(),
                "train/label_change_loss": loss_dict["loss_label_change"].item(),
                "train/text_preserve_loss": loss_dict["loss_preserve"].item(),
                "train/loss_angular": loss_dict["loss_angular"].item(),
                "train/loss_pull_away": loss_dict["loss_pull_away"].item(),
                "train/boundary_loss": loss_dict["loss_boundary"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/diversity": pos_neg_diff,
                "train/dynamic_scalar": model.module.combiner.get_newest(),
            },
        )

        del (
            img_emb,
            txt_full,
            comb_emb,
            loss,
            label_embedding,
        )

        torch.cuda.empty_cache()

    return epoch_metrics


def run(cfg: DictConfig, **kwargs):
    # Get args
    logger_dir = kwargs["logger_dir"]
    wandb_run = kwargs["wandb_run"]
    # Initialize Model
    model = CDC(
        clip_trainable=False,
        d_model=cfg.model.d_model,
        nhead=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        label_dim=cfg.model.label_dim,
    )
    model = nn.DataParallel(model)
    model.to(device)

    # Print model summary
    print_model_info(model.module.combiner)

    # preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize FolderManager
    folder_manager = FolderManager(base_log_dir=cfg.dataset.log_path)

    # Initialize feature manager
    feature_manager = FeatureManager(
        cfg.dataset.extract_path, chunk_size=cfg.train.batch_size
    )

    # Initialize experiment
    experiment_dir, init_dir, plot_dir = folder_manager.initialize_experiment(
        cfg.log_tag
    )
    checkpoint_dir, logs_dir = folder_manager.create_directories(experiment_dir)

    if cfg.dataset.pre_extract:
        print("##########Extracting and storing features##########")
        sample_ids_list = extract_and_store_features(
            cfg.dataset.train_path,
            cfg.dataset.img_path,
            feature_manager,
            cfg.train.batch_size,
            model,
            processor,
            device,
            ratio=cfg.dataset.ratio,
        )
        torch.save(
            sample_ids_list,
            os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"),
        )
    else:
        print("##########Loading pre-extracted features##########")
        sample_ids_list = torch.load(
            os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"),
            weights_only=False,
        )
        # turn sample_ids_list into a list of integers
        feature_manager.load_features()

    sample_ids_list = [int(sample_id) for sample_id in sample_ids_list]

    # Initialize embedding manager
    print("##########Initializing Embedding Manager##########")
    annotations = json.load(open(cfg.dataset.train_path))
    annotations = annotations[: int(len(annotations) * cfg.dataset.ratio)]
    embedding_manager = EmbeddingManager(
        annotations,
        embedding_dim=cfg.model.label_dim,
        chunk_size=cfg.train.batch_size,
        embeddings_dir=init_dir,
        sample_ids_list=sample_ids_list,
        use_template=USE_TEMPLATE,
    )
    embedding_manager.load_embeddings()

    # Samples to track
    samples_to_track = [0, 1, 2, 3, 4]  # Indices of the samples to track

    # Initialize clustering
    clustering = Clustering()

    # Create Train and Test dataloader
    train_dataset = CDC_train(
        annotation_path=cfg.dataset.train_path,
        image_path=cfg.dataset.img_path,
        embedding_manager=embedding_manager,
        feature_manager=feature_manager,
        ratio=cfg.dataset.ratio,
    )

    # batch_size = 1 and no shuffle, just load chunk embeddings
    train_dataloader = DataLoader(
        train_dataset,
        # batch_size=cfg.train.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    test_dataset = CDC_test(
        annotation_path=cfg.dataset.test_path,
        image_path=cfg.dataset.img_path_test,
        processor=processor,
        ratio=0.2 if "redcaps" in cfg.dataset.test_path else 1,
    )

    print(f"Test dataset size: {len(test_dataset)}")  # 0.2 for redcaps, 1 for flickr30k

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    unique_embeddings = None
    # Start training
    for epoch in range(max_epoch):
        logger_epoch = {}
        logger_epoch["epoch"] = epoch

        # Train
        if cfg.control.train:  # Network training
            print(f"##########Epoch {epoch}: Training##########")
            train_epoch_log = train(
                cfg,
                model=model,
                train_dataloader=train_dataloader,
                epoch=epoch,
                criteria=criteria,
                optimizer=optimizer,
                embedding_manager=embedding_manager,
                update_label_embedding=update_label_embedding,
                scheduler=scheduler,
                wandb_run=wandb_run,
            )

        if cfg.control.val:
            print("##########Testing train dataset##########")
            inf_train_log = inference_train(
                model, train_dataloader, device, epoch, [1, 5, 10]
            )
            wandb_run.log(inf_train_log)
            logger_epoch["inference_train"] = inf_train_log

        if cfg.control.train_2:  # KMeans update
            n_clusters = n_clusters_list[
                epoch
            ]  # Number of clusters for the current epoch
            # An adaptive alpha which minimum 0.1 and maximum 0.9, slide depends on k_means_middle_epoch - k_means_start_epoch
            alpha = max(
                min(
                    (1 - (k_means_middle_epoch - epoch) / k_means_middle_epoch),
                    alpha_upper,
                ),
                0.01,
            )

        if cfg.control.test:
            if unique_embeddings is not None:

                kmeans = KMeans(n_clusters=min(50, unique_embeddings.shape[0])).fit(
                    unique_embeddings.cpu().numpy()
                )
                centroids = kmeans.cluster_centers_
                # Find closest real embedding to each centroid

                indices = np.argmin(cdist(centroids, unique_embeddings), axis=1)
                representatives = unique_embeddings[indices]
                print("##########Testing test dataset##########")
                inf_test_log = inference_test(
                    model, processor, test_dataloader, representatives, epoch, device
                )
                logger_epoch["inference_test"] = inf_test_log
                wandb_run.log(inf_test_log)
                
        if cfg.control.save_per_epoch is True and unique_embeddings is not None:
            # Save model, epoch, optimizer, scheduler
            folder_manager.save_model(model, checkpoint_dir, epoch)
            cluster_folder = folder_manager.get_cluster_folder(experiment_dir)
            torch.save(
                unique_embeddings,
                os.path.join(cluster_folder, f"unique_embeddings_{epoch}.pt"),
            )

        # Save logger per epoch
        logger.append(logger_epoch)
        OmegaConf.save(config=cfg, f=os.path.join(experiment_dir, "config.yaml"))
        folder_manager.save_final_model(model.module.combiner, experiment_dir)

    # Save final model and merge history
    # folder_manager.save_final_model(model, experiment_dir)
    # OmegaConf.save(config=cfg, f=os.path.join(experiment_dir, "config.yaml"))

    # Clean cuda cache
    del (
        model,
        train_dataset,
        train_dataloader,
        criteria,
        optimizer,
        scheduler,
    )
    torch.cuda.empty_cache()

    return logger


@hydra.main(config_path="configs", config_name="redcaps", version_base=None)
def main(cfg):
    # Set seed
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize wandb
    project = cfg.wandb.project
    entity = cfg.wandb.entity
    tags = cfg.wandb.tags
    wandb.require("core")  # type: ignore
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_run = wandb.init(project=project, entity=entity, tags=tags, config=config_dict)  # type: ignore

    # Save a copy of config file
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_dir = f"logs/{now}_{cfg.log_tag}"
    os.mkdir(logger_dir)
    OmegaConf.save(config=cfg, f=os.path.join(logger_dir, "config.yaml"))
    # Print config
    print(OmegaConf.to_yaml(cfg))
    # Save config to wandb

    # Run main function
    logger = run(cfg=cfg, logger_dir=logger_dir, wandb_run=wandb_run)
    json.dump(logger, open(os.path.join(logger_dir, "logger.json"), "w"))

    wandb.finish()


if __name__ == "__main__":
    main()
