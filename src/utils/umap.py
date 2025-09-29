import matplotlib.pyplot as plt
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


def main():
    test_umap()


if __name__ == "__main__":
    main()
