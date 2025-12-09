import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np


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
    image_embs, text_embs, all_captions_flat, img_to_captions_map = precomputed_data

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
        top_k_values, top_k_indices = torch.topk(
            similarities, k=min(k, len(similarities))
        )

        top_k_captions = [
            all_captions_flat[idx] for idx in top_k_indices.cpu().tolist()
        ]

        return top_k_captions


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

    fig, axes = plt.subplots(3, 4, figsize=(36, 27))
    axes = axes.flatten()

    for i in range(n_bins):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])

        if not bin_mask.any():
            axes[i].axis("off")
            axes[i].set_title(f"Angle bin {i}: No data", fontsize=12)
            continue

        bin_conditions = conditions_2d[bin_mask]
        bin_center = bin_conditions.median(dim=0)[0]

        # ✅ 快速检索
        top_captions = retrieve_with_condition_fast(
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
            fontsize=9,
            va="top",
            ha="left",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
            transform=ax.transAxes,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=480, bbox_inches="tight")

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
    image_embs, text_embs, all_raw_image, all_raw_text = precomputed_data

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
        top_k_values, top_k_indices = torch.topk(
            similarities, k=min(k, len(similarities))
        )

        # Retrieve top-k images
        top_k_images = [all_raw_image[idx] for idx in top_k_indices.cpu().tolist()]

        return top_k_images


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

    fig, axes = plt.subplots(3, 4, figsize=(36, 27))
    axes = axes.flatten()

    for i in range(n_bins):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])

        if not bin_mask.any():
            axes[i].axis("off")
            axes[i].set_title(f"Angle bin {i}: No data", fontsize=12)
            continue

        bin_conditions = conditions_2d[bin_mask]
        bin_center = bin_conditions.median(dim=0)[0]

        # Fast retrieval: text -> images
        top_images = retrieve_image_with_condition_fast(
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
        ax.set_title(title, fontsize=10, fontweight="bold")

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
