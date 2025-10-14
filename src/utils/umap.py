import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from scipy import stats


def print_model_info(model):
    # 1. Print the structure of the model
    print("Model Architecture:\n")
    print(model)

    # 2. Count the total number of parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nTotal number of parameters: ", total_params)
    print("Number of trainable parameters: ", trainable_params)


def get_umap(
    umap_features_np,
    umap_labels,
    epoch,
    samples_to_track=[],
    z_threshold=3,
    no_outlier=True,
):

    fig = plt.figure(figsize=(10, 10))

    if umap_labels is None:
        umap_labels = np.ones_like(umap_features_np[:, 0])

    if no_outlier:
        # Compute the z-scores of the features for each dimension
        z_scores = np.abs(stats.zscore(umap_features_np, axis=0))

        # Filter the points by applying a z-score threshold (default = 3)
        non_outliers = (z_scores < z_threshold).all(axis=1)

        # Apply the outlier filter
        filtered_features = umap_features_np[non_outliers]
        filtered_labels = umap_labels[non_outliers]

    else:
        filtered_features = umap_features_np
        filtered_labels = umap_labels

    tmp_labels = filtered_labels >= 0

    plt.scatter(
        filtered_features[~tmp_labels, 0],
        filtered_features[~tmp_labels, 1],
        color=[0.5, 0.5, 0.5],
        s=0.5,
        alpha=0.75,
    )

    plt.scatter(
        filtered_features[tmp_labels, 0],
        filtered_features[tmp_labels, 1],
        c=filtered_labels[tmp_labels],
        s=0.5,
        alpha=0.75,
    )

    # Highlight and annotate the tracked samples (if they are not outliers)
    if len(samples_to_track) > 0:
        for sample_idx in samples_to_track:
            if (
                not no_outlier or non_outliers[sample_idx]
            ):  # Only track samples that are not outliers
                x, y = umap_features_np[sample_idx, :]
                plt.scatter(x, y, c="red", s=50, edgecolors="k")  # Highlight the sample
                plt.text(
                    x, y, f"Sample {sample_idx}", fontsize=12, color="black"
                )  # Annotate the sample

    # Add the number of UMAP labels to the plot as title
    plt.title(
        f"UMAP clusters at epoch {epoch} (outliers removed)"
        if no_outlier
        else f"UMAP clusters at epoch {epoch}"
    )
    plt.colorbar()

    return fig


def visualize_ideal_condition_space(conditions_2d, epoch):
    """
    理想的可视化应该显示：
    - 清晰的径向结构（同心圆）
    - 角度方向上的平滑过渡
    - 8-12个明显的聚类区域
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ✅ 修复：统一转换为numpy array
    if isinstance(conditions_2d, torch.Tensor):
        conditions_np = conditions_2d.detach().cpu().numpy()
    else:
        conditions_np = np.array(conditions_2d)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={"projection": "polar"})
    axes = axes.flatten()  # 将2x2的axes展平为1D数组

    # 转换为极坐标
    angles = np.arctan2(conditions_np[:, 1], conditions_np[:, 0])
    radii = np.linalg.norm(conditions_np, axis=1)

    # 图1: 真正的极坐标散点图
    ax = axes[0]
    ax.scatter(angles, radii, alpha=0.6, s=8)
    ax.set_title(f"Polar Coordinate Visualization at epoch {epoch}")
    ax.set_ylim(0, np.max(radii) * 1.1)

    # 添加角度标记
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(
        [f"{int(np.degrees(t))}°" for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    )

    # 图2: 笛卡尔坐标系中的散点图 + 极坐标网格
    ax = axes[1]
    ax.remove()  # 移除极坐标投影
    ax = fig.add_subplot(2, 2, 2)  # 添加新的笛卡尔坐标轴

    ax.scatter(conditions_np[:, 0], conditions_np[:, 1], alpha=0.6, s=8)

    # 添加极坐标网格
    max_radius = np.max(radii)
    for r in np.linspace(0.2, max_radius, 5):
        circle = patches.Circle(
            (0, 0), r, fill=False, linestyle="--", alpha=0.3, color="gray"
        )
        ax.add_patch(circle)

    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot(
            [0, max_radius * np.cos(angle)],
            [0, max_radius * np.sin(angle)],
            "k--",
            alpha=0.3,
            linewidth=0.5,
        )

    ax.set_aspect("equal")
    ax.set_title(f"Cartesian with Polar Grid at epoch {epoch}")
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")

    # 图3: 按半径着色
    ax = axes[2]
    ax.remove()  # 移除极坐标投影
    ax = fig.add_subplot(2, 2, 3)  # 添加新的笛卡尔坐标轴

    scatter = ax.scatter(
        conditions_np[:, 0],
        conditions_np[:, 1],
        c=radii,
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter, ax=ax, label="Radius (Modulation Strength)")
    ax.set_aspect("equal")
    ax.set_title(f"Colored by Radius at epoch {epoch}")
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")

    # 图4: 按角度着色
    ax = axes[3]
    ax.remove()  # 移除极坐标投影
    ax = fig.add_subplot(2, 2, 4)  # 添加新的笛卡尔坐标轴

    scatter = ax.scatter(
        conditions_np[:, 0], conditions_np[:, 1], c=angles, cmap="hsv", alpha=0.6, s=10
    )
    plt.colorbar(scatter, ax=ax, label="Angle (Semantic Type)")
    ax.set_aspect("equal")
    ax.set_title(f"Colored by Angle at epoch {epoch}")
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")

    plt.tight_layout()

    return fig


def visualize_angular_semantics(conditions_2d, model, test_set):
    """
    显示每个角度区间的典型检索结果
    理想效果：清晰的语义渐变
    """
    n_bins = 12
    angles = torch.atan2(conditions_2d[:, 1], conditions_2d[:, 0])
    angle_bins = torch.linspace(-np.pi, np.pi, n_bins + 1)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(n_bins):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        bin_conditions = conditions_2d[bin_mask]

        if len(bin_conditions) == 0:
            continue

        # 选择该bin的中位数condition
        bin_center = bin_conditions.median(0)[0]

        # 检索 (需要实现 retrieve_with_condition 函数)
        # top_captions = retrieve_with_condition(
        #     model, test_set, bin_center, k=5
        # )  # 更新方程
        top_captions = [f"Caption {j+1}" for j in range(5)]  # 占位符

        # 显示
        ax = axes[i]
        ax.axis("off")
        ax.set_title(
            f"Angle: {np.degrees(angle_bins[i]):.0f}° - {np.degrees(angle_bins[i+1]):.0f}°"
        )

        # 显示检索到的captions
        text = "\n".join([f"{j+1}. {cap}" for j, cap in enumerate(top_captions)])
        ax.text(0.1, 0.5, text, fontsize=8, va="center", wrap=True)

    plt.tight_layout()

    return fig


def test_umap():
    umap_features_np = np.random.rand(10000, 2)
    umap_labels = torch.tensor([0] * 5000 + [1] * 5000, device="cpu")
    print(umap_labels.shape)

    fig = get_umap(
        umap_features_np,
        umap_labels,
        epoch=0,
        no_outlier=False,
        samples_to_track=[0, 1, 2, 3, 4],
    )
    fig.savefig("src/utils/umap/umap.png", dpi=480, bbox_inches="tight")

    fig = get_umap(
        umap_features_np,
        umap_labels,
        epoch=0,
        no_outlier=True,
        samples_to_track=[0, 1, 2, 3, 4],
    )
    fig.savefig("src/utils/umap/umap_nooutlier.png", dpi=480, bbox_inches="tight")


def test_ideal_condition_space():
    conditions_2d = np.random.rand(1000, 2)
    fig = visualize_ideal_condition_space(conditions_2d, epoch=0)
    fig.savefig(
        "src/utils/umap/ideal_condition_space.png", dpi=480, bbox_inches="tight"
    )


def main():
    test_umap()
    test_ideal_condition_space()


if __name__ == "__main__":
    main()
