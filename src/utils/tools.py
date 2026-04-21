import textwrap
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time


def replace_with_most_different(label_embeddings, k=10):
    batch_size = label_embeddings.size(0)

    # Normalize the embeddings
    normalized = F.normalize(label_embeddings, p=2, dim=1)

    # Compute cosine similarity and distance
    cosine_sim = torch.matmul(normalized, normalized.T)
    cosine_dist = 1 - cosine_sim

    # Fill diagonal with large negative to avoid self-match
    cosine_dist.fill_diagonal_(-float("inf"))

    # Get indices of top-k most dissimilar embeddings
    topk_indices = torch.topk(cosine_dist, k=k, dim=1).indices  # [B, k]

    # Sample one index from top-k for each row
    rand_indices = torch.randint(0, k, (batch_size,), device=label_embeddings.device)
    selected_indices = topk_indices[torch.arange(batch_size), rand_indices]

    # Gather new embeddings
    new_embeddings = label_embeddings[selected_indices]

    return new_embeddings


def precompute_nearest_condition_labels(
    train_conditions: torch.Tensor,
    selected_conditions: torch.Tensor,
    print_statistics: bool = False,
) -> torch.Tensor:
    """
    For each training sample, find the nearest selected condition.

    Args:
        train_conditions: Phase 1 conditions for training set [N, 2]
        selected_conditions: K selected conditions [K, 2]

    Returns:
        labels: Index of nearest condition for each sample [N]
    """

    # Compute pairwise distances
    # [N, 1, 2] - [1, K, 2] → [N, K]
    dists = torch.norm(
        train_conditions.unsqueeze(1) - selected_conditions.unsqueeze(0), dim=2
    )

    # Find nearest
    labels = dists.argmin(dim=1)  # [N]

    if print_statistics:
        # Statistics
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        for idx, count in zip(unique.tolist(), counts.tolist()):
            print(f"  Condition {idx}: {count} samples ({count/len(labels)*100:.1f}%)")

    return labels


def select_representative_conditions(
    all_conditions: torch.Tensor,
    K: int = 12,
    method: str = "kmeans",
    seed: int = 42,
) -> torch.Tensor:
    """
    Select K representative conditions from Phase 1.

    Args:
        all_conditions: All Phase 1 conditions [N, 2]
        K: Number of conditions to select
        method: Selection method ('kmeans', 'diverse', 'random')
        seed: Random seed

    Returns:
        selected_conditions: K selected conditions [K, 2]
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    all_conditions_np = all_conditions.cpu().numpy()

    if method == "kmeans":
        print(f"Selecting {K} conditions using K-means clustering...")
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
        kmeans.fit(all_conditions_np)

        selected = torch.from_numpy(kmeans.cluster_centers_).float()

    elif method == "diverse":
        print(f"Selecting {K} conditions using diversity sampling...")

        # Greedy selection: maximize minimum distance
        selected_indices = []

        # Start with random point
        selected_indices.append(np.random.randint(len(all_conditions)))

        for _ in range(K - 1):
            # Compute distances to selected points
            selected_points = all_conditions_np[selected_indices]
            dists = np.min(
                np.linalg.norm(
                    all_conditions_np[:, None] - selected_points[None, :], axis=2
                ),
                axis=1,
            )

            # Select point with maximum distance
            next_idx = np.argmax(dists)
            selected_indices.append(next_idx)

        selected = all_conditions[selected_indices]

    elif method == "random":
        print(f"Selecting {K} conditions randomly...")
        indices = torch.randperm(len(all_conditions))[:K]
        selected = all_conditions[indices]

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Selected {K} conditions:")
    print(f"  Mean: {selected.mean(dim=0)}")
    print(f"  Std:  {selected.std(dim=0)}")
    print(f"  Min:  {selected.min(dim=0).values}")
    print(f"  Max:  {selected.max(dim=0).values}")

    return selected


def get_representatives(label_embeddings, k=10):
    # Check if label_embeddings are with dimention 2
    assert label_embeddings.shape[1] == 2, "Label embeddings must be with dimention 2"
    kmeans = KMeans(n_clusters=min(k, label_embeddings.shape[0])).fit(
        label_embeddings.cpu().numpy()
    )
    centroids = kmeans.cluster_centers_
    # Find closest real embedding to each centroid
    indices = np.argmin(cdist(centroids, label_embeddings), axis=1)
    representatives = label_embeddings[indices]
    return representatives


def get_representatives_polar_grid(
    learned_conditions=None, num_angles=10, num_radii=3, max_radius=None
):
    """
    在极坐标下均匀采样
    Args:
        num_angles: 角度方向数 (m)
        num_radii: 半径层数 (n)
        max_radius: 最大半径，如果None则从learned_conditions推断
        learned_conditions: [N, 2] 训练学到的conditions
    Returns:
        sampled_conditions: [k, 2] where k = num_angles * num_radii
    """
    if max_radius is None and learned_conditions is not None:
        # 从learned conditions推断合理的radius范围
        radii = torch.norm(learned_conditions, dim=1)
        max_radius = radii.quantile(0.95)  # 用95分位数，避免outlier
        print(f"Inferred max_radius: {max_radius:.3f}")
    elif max_radius is None:
        max_radius = 2.0  # 默认值

    # 生成角度: [0, 2π) 均匀分布
    angles = torch.linspace(0, 2 * torch.pi, num_angles + 1)[:-1]  # 不包括2π

    # 生成半径: [0, max_radius] 均匀或对数分布
    # 选项1: 线性分布（包括0）
    radii = torch.linspace(0, max_radius, num_radii)

    # 选项2: 不包括0（如果你发现0附近的condition不重要）
    # radii = torch.linspace(0.1 * max_radius, max_radius, num_radii)

    # 生成网格
    sampled_conditions = []
    for r in radii:
        for theta in angles:
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            sampled_conditions.append([x.item(), y.item()])

    sampled_conditions = torch.tensor(sampled_conditions)
    print(
        f"Sampled {len(sampled_conditions)} conditions "
        f"({num_angles} angles × {num_radii} radii)"
    )

    return sampled_conditions


def get_representatives_fps(
    conditions: torch.Tensor, k: int = 10, inlier_quantile: float = 0.95
) -> torch.Tensor:
    """Farthest-point sampling: returns k maximally spread representatives from conditions [N, 2].

    Filters out outliers beyond `inlier_quantile` of L2 distance from the centroid
    before sampling, so extreme outliers don't anchor the selection.
    """
    start_time = time.time()
    centroid = conditions.mean(dim=0)
    dists_from_centroid = torch.norm(conditions - centroid, dim=1)
    radius = dists_from_centroid.quantile(inlier_quantile)
    inliers = conditions[dists_from_centroid <= radius]

    k = min(k, len(inliers))
    selected = [torch.randint(len(inliers), (1,)).item()]
    for _ in range(k - 1):
        dists = torch.cdist(inliers, inliers[selected]).min(dim=1).values
        selected.append(dists.argmax().item())
    print(f"Time taken to get representatives: {time.time() - start_time:.3f}s")
    return inliers[selected]


def get_representatives_polar_grid_outsideonly(learned_conditions, num_angles=10):
    """
    For each angular sector, pick the real learned condition closest to the 0.95
    quantile radius within that sector. Returns one real sample per sector.

    Args:
        learned_conditions: [N, 2] learned conditions
        num_angles: number of angular sectors
    Returns:
        sampled_conditions: [num_angles, 2]
    """
    angles = torch.atan2(learned_conditions[:, 1], learned_conditions[:, 0]) % (
        2 * torch.pi
    )
    radii = torch.norm(learned_conditions, dim=1)

    sector_size = 2 * torch.pi / num_angles
    sector_ids = (angles / sector_size).long().clamp(0, num_angles - 1)

    sampled = []
    for i in range(num_angles):
        mask = sector_ids == i
        if mask.sum() == 0:
            # No points in this sector: fall back to closest point by angle
            center = (i + 0.5) * sector_size
            diffs = (angles - center + torch.pi) % (2 * torch.pi) - torch.pi
            mask = torch.zeros(len(angles), dtype=torch.bool)
            mask[diffs.abs().argmin()] = True

        pts = learned_conditions[mask]
        r = radii[mask]
        q = r.quantile(0.95)
        idx = (r - q).abs().argmin()
        sampled.append(pts[idx])

    result = torch.stack(sampled)
    print(
        f"Sampled {len(result)} conditions ({num_angles} angular sectors, 0.95 quantile radius per sector)"
    )
    return result


def precompute_coco_embeddings(model, coco_test_loader, device, processor):
    """
    预计算所有embeddings，加速后续检索

    Returns:
        image_embs: [N_images, 512]
        text_embs: [N_images*5, 512]
        captions_flat: list of all captions
        image_to_captions_map: dict mapping image_idx to caption indices
    """
    model.eval()

    all_image_embs = []
    all_text_embs = []
    all_captions_flat = []
    image_to_captions_map = {}

    caption_idx = 0

    max_text_length = 77

    with torch.no_grad():
        for batch_idx, batch in enumerate(coco_test_loader):
            image_input, captions = batch
            image_input = image_input.to(device)

            # Image embeddings
            img_embs, _ = model.encode_img(image_input)
            all_image_embs.append(img_embs)

            # Text embeddings
            batch_size = len(image_input)
            for i in range(batch_size):
                # 这张图的5个captions
                if isinstance(captions[i], list):
                    img_captions = captions[i]
                else:
                    # 如果是flat的
                    img_captions = [captions[i * 5 + j] for j in range(5)]

                print(captions[i + 1])

                # 记录映射
                img_idx = batch_idx * batch_size + i
                image_to_captions_map[img_idx] = list(
                    range(caption_idx, caption_idx + 5)
                )

                # Encode captions
                text_tokens = processor(
                    text=img_captions,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_text_length,
                ).to(device)
                text_embs, _ = model.encode_txt(text_tokens)

                all_text_embs.append(text_embs)
                all_captions_flat.extend(img_captions)

                caption_idx += 5

    # 合并
    image_embs = torch.cat(all_image_embs, dim=0)  # [N_images, 512]
    text_embs = torch.cat(all_text_embs, dim=0)  # [N_images*5, 512]

    return image_embs, text_embs, all_captions_flat, image_to_captions_map


def retrieve_with_condition_fast(
    model, precomputed_data, condition, query_image_id=None, k=10, device=None
):
    """
    使用预计算的embeddings快速检索
    """
    image_embs, text_embs, all_captions_flat, _ = precomputed_data

    model.eval()
    # 处理condition
    if isinstance(condition, np.ndarray):
        condition = torch.from_numpy(condition).float()
    if condition.dim() == 1:
        condition = condition.unsqueeze(0)
    condition = condition.to(device)

    with torch.no_grad():
        # 选择query
        if query_image_id is None:
            query_image_id = torch.randint(0, len(image_embs), (1,)).item()

        img_emb = image_embs[query_image_id : query_image_id + 1]  # [1, 512]

        # 调制所有text embeddings
        condition_expanded = condition.repeat(len(text_embs), 1)
        text_embs = text_embs.to(device)
        condition_expanded = condition_expanded.to(device)
        img_emb = img_emb.to(device)
        text_embs_modulated = model.combine(text_embs, None, condition_expanded)

        # 相似度
        similarities = F.cosine_similarity(
            img_emb.expand(len(text_embs_modulated), -1), text_embs_modulated, dim=1
        )

        # Top-k
        _, top_k_indices = torch.topk(
            similarities, k=min(k, len(similarities))
        )

        idx_list = top_k_indices.cpu().tolist()
        top_k_captions = [all_captions_flat[idx] for idx in idx_list]

        return idx_list, top_k_captions


def visualize_angular_semantics_fast(
    conditions_2d, model, precomputed_data, save_path=None, device=None
):
    """
    使用预计算embeddings的快速可视化
    """

    model.eval()

    if isinstance(conditions_2d, np.ndarray):
        conditions_2d = torch.from_numpy(conditions_2d).float()
    conditions_2d = conditions_2d.to(device)

    angles = torch.atan2(conditions_2d[:, 1], conditions_2d[:, 0])

    n_bins = 12
    angle_bins = torch.linspace(-np.pi, np.pi, n_bins + 1, device=device)

    # 固定query image
    fixed_query_id = torch.randint(0, len(precomputed_data[0]), (1,)).item()

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()

    for i in range(n_bins):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])

        if not bin_mask.any():
            axes[i].axis("off")
            axes[i].set_title(f"Angle bin {i}: No data", fontsize=8)
            continue

        bin_conditions = conditions_2d[bin_mask]
        bin_center = bin_conditions.median(dim=0)[0]

        # ✅ 快速检索
        _, top_captions = retrieve_with_condition_fast(
            model,
            precomputed_data,
            bin_center,
            query_image_id=fixed_query_id,
            k=20,
            device=device,
        )

        # 显示（同上）
        ax = axes[i]
        ax.axis("off")

        title = f"Angle: {np.degrees(angle_bins[i].item()):.0f}° - {np.degrees(angle_bins[i+1].item()):.0f}°"
        ax.set_title(title, fontsize=11, fontweight="bold")

        text_content = "\n\n".join(
            [f"{j+1}. {cap[:100]}" for j, cap in enumerate(top_captions)]
        )
        ax.text(
            0.02,
            0.98,
            text_content,
            fontsize=6,
            va="top",
            ha="left",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
            transform=ax.transAxes,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def retrieve_image_with_condition_fast(
    model, precomputed_data, condition, query_text_id=None, k=5, device=None
):
    """
    Text-to-image retrieval: fix a text, retrieve top-k images with condition modulation

    Args:
        model: CoSiR model
        precomputed_data: tuple of (all_img_emb, all_txt_emb, all_raw_image)
        condition: condition vector [2] or [1, 2]
        query_text_id: fixed text index, if None, randomly select one
        k: number of top images to retrieve
        device: computation device

    Returns:
        top_k_images: list of top-k retrieved images (PIL Images or image data)
    """
    image_embs, text_embs, all_raw_image, _ = precomputed_data

    model.eval()

    # Process condition
    if isinstance(condition, np.ndarray):
        condition = torch.from_numpy(condition).float()
    if condition.dim() == 1:
        condition = condition.unsqueeze(0)
    condition = condition.to(device)

    with torch.no_grad():
        # Select query text
        if query_text_id is None:
            query_text_id = torch.randint(0, len(text_embs), (1,)).item()

        txt_emb = text_embs[query_text_id : query_text_id + 1]  # [1, 512]

        # Modulate the query text embedding with condition
        txt_emb = txt_emb.to(device)
        condition = condition.to(device)
        txt_emb_modulated = model.combine(txt_emb, None, condition)

        # Compute similarity with all images
        image_embs = image_embs.to(device)
        similarities = F.cosine_similarity(
            txt_emb_modulated.expand(len(image_embs), -1), image_embs, dim=1
        )

        # Top-k
        _, top_k_indices = torch.topk(
            similarities, k=min(k, len(similarities))
        )

        idx_list = top_k_indices.cpu().tolist()
        top_k_images = [all_raw_image[idx] for idx in idx_list]

        return idx_list, top_k_images


def visualize_angular_semantics_text_to_image_fast(
    conditions_2d, model, precomputed_data, save_path=None, device=None
):
    """
    Text-to-image version: visualize how different condition angles affect image retrieval
    for a fixed text query

    Args:
        conditions_2d: [N, 2] condition embeddings
        model: CoSiR model
        precomputed_data: tuple of (all_img_emb, all_txt_emb, all_raw_text, all_raw_image, all_raw_text)
        save_path: optional path to save the figure
        device: computation device

    Returns:
        fig: matplotlib figure
    """
    model.eval()

    if isinstance(conditions_2d, np.ndarray):
        conditions_2d = torch.from_numpy(conditions_2d).float()
    conditions_2d = conditions_2d.to(device)

    angles = torch.atan2(conditions_2d[:, 1], conditions_2d[:, 0])

    n_bins = 12
    angle_bins = torch.linspace(-np.pi, np.pi, n_bins + 1, device=device)

    # Fix query text
    fixed_query_text_id = torch.randint(0, len(precomputed_data[1]), (1,)).item()
    query_text = precomputed_data[3][fixed_query_text_id]

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()

    for i in range(n_bins):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])

        if not bin_mask.any():
            axes[i].axis("off")
            axes[i].set_title(f"Angle bin {i}: No data", fontsize=8)
            continue

        bin_conditions = conditions_2d[bin_mask]
        bin_center = bin_conditions.median(dim=0)[0]

        # Fast retrieval: text -> images
        _, top_images = retrieve_image_with_condition_fast(
            model,
            precomputed_data,
            bin_center,
            query_text_id=fixed_query_text_id,
            k=9,  # 3x3 grid of images
            device=device,
        )

        # Display images in grid
        ax = axes[i]
        ax.axis("off")

        title = f"Angle: {np.degrees(angle_bins[i].item()):.0f}° - {np.degrees(angle_bins[i+1].item()):.0f}°\nQuery: {query_text[:60]}..."
        ax.set_title(title, fontsize=7, fontweight="bold")

        # Create a 3x3 grid of images within this subplot
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        inner_grid = GridSpecFromSubplotSpec(
            3, 3, subplot_spec=ax.get_subplotspec(), wspace=0.02, hspace=0.02
        )

        for img_idx, img in enumerate(top_images):
            if img_idx >= 9:  # Only show 3x3 grid
                break
            row = img_idx // 3
            col = img_idx % 3
            inner_ax = plt.subplot(inner_grid[row, col])
            inner_ax.imshow(img)
            inner_ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=480, bbox_inches="tight")

    return fig


def _clip_baseline_i2t(image_embs, text_embs, all_captions_flat, query_id, k, device):
    """Raw CLIP image→text retrieval without condition modulation."""
    img_emb = image_embs[query_id : query_id + 1].to(device)
    txt = text_embs.to(device)
    sims = F.cosine_similarity(img_emb.expand(len(txt), -1), txt, dim=1)
    _, top_idx = torch.topk(sims, k=min(k, len(sims)))
    idx_list = top_idx.cpu().tolist()
    return idx_list, [all_captions_flat[i] for i in idx_list]


def _clip_baseline_t2i(image_embs, text_embs, all_raw_image, query_text_id, k, device):
    """Raw CLIP text→image retrieval without condition modulation."""
    txt_emb = text_embs[query_text_id : query_text_id + 1].to(device)
    imgs = image_embs.to(device)
    sims = F.cosine_similarity(txt_emb.expand(len(imgs), -1), imgs, dim=1)
    _, top_idx = torch.topk(sims, k=min(k, len(sims)))
    idx_list = top_idx.cpu().tolist()
    return idx_list, [all_raw_image[i] for i in idx_list]


def visualize_given_conditions_image_to_text(
    conditions, model, precomputed_data, n_queries=3, k=5, all_raw_image=None, device=None
):
    """
    Image-to-text retrieval visualization with pre-given conditions.

    Row 0 = CLIP baseline (no condition). Rows 1..N = one row per condition.
    Query image spans all rows on the left when all_raw_image is provided.
    Green cells = ground-truth captions.

    Args:
        conditions: [N, D] condition embeddings (pre-sampled)
        model: CoSiR model
        precomputed_data: (all_img_emb, all_txt_emb, all_captions_flat, img_to_cap_map)
        n_queries: number of query images (uses first n_queries)
        k: top-k captions per row
        all_raw_image: optional list of PIL images for showing the query
        device: computation device

    Returns:
        list of figures, one per query image
    """
    model.eval()

    if isinstance(conditions, np.ndarray):
        conditions = torch.from_numpy(conditions).float()
    conditions = conditions.to(device)

    image_embs, text_embs, all_captions_flat, img_to_cap_map = precomputed_data
    n_conditions = len(conditions)
    n_rows = n_conditions + 1          # +1 for CLIP baseline row
    show_image = all_raw_image is not None

    img_col_w = min(2.0 + n_rows * 0.12, 3.5)   # scales with total rows
    txt_col_w = 2.6
    row_h = 1.5
    n_cols = k + (1 if show_image else 0)
    total_w = (img_col_w if show_image else 0) + txt_col_w * k

    figs = []

    for query_id in range(n_queries):
        gt_raw = img_to_cap_map[query_id]
        gt_cap_set = set(gt_raw.tolist() if isinstance(gt_raw, torch.Tensor) else gt_raw)

        fig = plt.figure(
            figsize=(total_w, row_h * n_rows),
            constrained_layout=True,
            dpi=80,
        )
        width_ratios = ([img_col_w] if show_image else []) + [txt_col_w] * k
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios)

        fig.suptitle(
            f"Image→Text  |  Query image idx={query_id}",
            fontsize=9, fontweight="bold",
        )

        # Query image spanning all rows (shown once)
        if show_image:
            ax_img = fig.add_subplot(gs[:, 0])
            ax_img.imshow(all_raw_image[query_id])
            ax_img.axis("off")
            ax_img.set_title("Query", fontsize=7, pad=2)

        txt_offset = 1 if show_image else 0

        def _render_row(row_idx, row_label, cap_indices, captions):
            for rank, (cap_idx, caption) in enumerate(zip(cap_indices, captions)):
                ax = fig.add_subplot(gs[row_idx, rank + txt_offset])
                ax.axis("off")

                if rank == 0:
                    ax.set_ylabel(
                        row_label, fontsize=7, rotation=0,
                        ha="right", va="center", labelpad=3,
                    )
                    ax.yaxis.set_label_coords(-0.06, 0.5)
                    ax.yaxis.label.set_visible(True)

                if row_idx == 0:
                    ax.set_title(f"Rank {rank + 1}", fontsize=7, pad=2)

                words = caption.split()
                truncated = " ".join(words[:30]) + ("…" if len(words) > 30 else "")
                wrapped = "\n".join(textwrap.wrap(truncated, width=38))
                facecolor = "limegreen" if cap_idx in gt_cap_set else "lightblue"

                ax.text(
                    0.5, 0.5, wrapped,
                    fontsize=5.5, va="center", ha="center",
                    family="monospace", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolor, alpha=0.35),
                    transform=ax.transAxes,
                )

        # Row 0: CLIP baseline
        base_idx, base_caps = _clip_baseline_i2t(
            image_embs, text_embs, all_captions_flat, query_id, k, device
        )
        _render_row(0, "CLIP", base_idx, base_caps)

        # Rows 1..N: conditioned
        for cond_idx, condition in enumerate(conditions):
            cap_indices, top_captions = retrieve_with_condition_fast(
                model, precomputed_data, condition,
                query_image_id=query_id, k=k, device=device,
            )
            _render_row(cond_idx + 1, f"C{cond_idx}", cap_indices, top_captions)

        figs.append(fig)

    return figs


def visualize_given_conditions_text_to_image(
    conditions, model, precomputed_data, n_queries=3, texts_per_image=5, k=4, device=None
):
    """
    Text-to-image retrieval visualization with pre-given conditions.

    Each figure corresponds to one query text and shows how different conditions
    change the retrieved images. Layout: N_conditions rows x k columns (images).

    The first n_queries * texts_per_image texts are used as queries (matching the
    first n_queries images' captions). Produces one figure per query text.

    Args:
        conditions: [N, D] condition embeddings (pre-sampled, not generated here)
        model: CoSiR model
        precomputed_data: tuple of (all_img_emb, all_txt_emb, all_raw_image, all_raw_text)
        n_queries: number of source images — queries are the first n_queries * texts_per_image texts
        texts_per_image: captions per image (default 5 for COCO-style datasets)
        k: number of top images to retrieve per condition
        device: computation device

    Returns:
        figs: list of matplotlib figures, one per query text
    """
    model.eval()

    if isinstance(conditions, np.ndarray):
        conditions = torch.from_numpy(conditions).float()
    conditions = conditions.to(device)

    n_conditions = len(conditions)
    image_embs, text_embs, all_raw_image, all_raw_text = precomputed_data
    n_rows = n_conditions + 1          # +1 for CLIP baseline row

    n_text_queries = n_queries * texts_per_image
    img_size = 1.5  # inches per image cell
    figs = []

    for query_text_id in range(n_text_queries):
        query_text = all_raw_text[query_text_id]
        source_image_idx = query_text_id // texts_per_image
        caption_idx = query_text_id % texts_per_image

        fig, axes = plt.subplots(
            n_rows, k,
            figsize=(img_size * k, img_size * n_rows),
            squeeze=False,
            dpi=80,
        )
        fig.suptitle(
            f"Text→Image  |  Src img {source_image_idx}, caption {caption_idx}\n"
            f'"{query_text[:90]}{"…" if len(query_text) > 90 else ""}"',
            fontsize=8, fontweight="bold", y=1.02,
        )

        def _render_img_row(row_idx, row_label, img_indices, images):
            for rank, (img_idx, img) in enumerate(zip(img_indices, images)):
                ax = axes[row_idx, rank]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                if rank == 0:
                    ax.set_ylabel(
                        row_label, fontsize=5.5, rotation=0, labelpad=4,
                        ha="right", va="center",
                    )
                    ax.yaxis.set_label_coords(-0.02, 0.5)
                    ax.yaxis.label.set_visible(True)

                if row_idx == 0:
                    ax.set_title(f"Rank {rank + 1}", fontsize=7, pad=2)

                is_gt = (img_idx == source_image_idx)
                for spine in ax.spines.values():
                    spine.set_visible(is_gt)
                    spine.set_edgecolor("limegreen" if is_gt else "none")
                    spine.set_linewidth(3 if is_gt else 0)

        # Row 0: CLIP baseline
        base_idx, base_imgs = _clip_baseline_t2i(
            image_embs, text_embs, all_raw_image, query_text_id, k, device
        )
        _render_img_row(0, "CLIP", base_idx, base_imgs)

        # Rows 1..N: conditioned
        for cond_idx, condition in enumerate(conditions):
            img_indices, top_images = retrieve_image_with_condition_fast(
                model, precomputed_data, condition,
                query_text_id=query_text_id, k=k, device=device,
            )
            _render_img_row(cond_idx + 1, f"C{cond_idx}", img_indices, top_images)

        plt.tight_layout()
        figs.append(fig)

    return figs
