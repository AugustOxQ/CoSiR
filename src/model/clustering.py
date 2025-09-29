# import numpy as np
import random
from collections import defaultdict
from cProfile import label

import dask.array as da
import numpy as np
import torch
from cuml.cluster import HDBSCAN, KMeans
from cuml.dask.manifold import UMAP as MNMG_UMAP
from cuml.datasets import make_blobs
from cuml.manifold import UMAP
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from scipy import cluster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UMAP_vis:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cluster = None
        self.client = None
        self.local_model = None  # Cache the UMAP model
        self._cluster_initialized = False

    def _ensure_cluster(self):
        # Lazy initialization of cluster
        if not self._cluster_initialized:
            self.cluster = LocalCUDACluster(threads_per_worker=1)
            self.client = Client(self.cluster)
            self._cluster_initialized = True

    def close_cluster(self):
        # Close Dask CUDA cluster and client
        if self._cluster_initialized:
            self.client.close()  # type: ignore
            self.cluster.close()  # type: ignore
            self._cluster_initialized = False
            self.local_model = None

    def learn_umap(self, embedding, n_components: int = 2):
        # Perform UMAP dimensionality reduction on embeddings
        self._ensure_cluster()

        # Optimize tensor operations - avoid unnecessary device transfers
        if embedding.device != self.device:
            embedding = embedding.to(self.device)
        label_embedding_np = embedding.cpu().numpy()

        # Create and cache UMAP model
        self.local_model = UMAP(random_state=42, n_components=n_components)
        umap_features = self.local_model.fit_transform(label_embedding_np)

        return torch.tensor(umap_features, device=self.device, dtype=embedding.dtype)

    def predict_umap(self, new_embedding):
        self._ensure_cluster()
        if self.local_model is None:
            raise ValueError("UMAP model has not been trained yet.")
        
        # Handle both numpy arrays and torch tensors
        if isinstance(new_embedding, torch.Tensor):
            if new_embedding.device != self.device:
                new_embedding = new_embedding.to(self.device)
            new_embedding_np = new_embedding.cpu().numpy()
            original_dtype = new_embedding.dtype
        else:
            new_embedding_np = new_embedding
            original_dtype = torch.float32  # Default dtype for numpy arrays
        
        umap_features_new = self.local_model.transform(new_embedding_np)
        return torch.tensor(umap_features_new, device=self.device, dtype=original_dtype)


class Clustering:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cluster = None
        self.client = None
        self._umap_model = None  # Cache for UMAP model
        self._kmeans_model = None  # Cache for KMeans model
        self._cluster_initialized = False

    def _ensure_cluster(self):
        # Lazy initialization of cluster
        if not self._cluster_initialized:
            self.cluster = LocalCUDACluster(threads_per_worker=1)
            self.client = Client(self.cluster)
            self._cluster_initialized = True

    def close_cluster(self):
        # Close Dask CUDA cluster and client
        if self._cluster_initialized:
            self.client.close()  # type: ignore
            self.cluster.close()  # type: ignore
            self._cluster_initialized = False
            self._umap_model = None
            self._kmeans_model = None

    def get_umap(self, label_embedding, n_components: int = 2):
        # Perform UMAP dimensionality reduction on embeddings
        self._ensure_cluster()

        # Optimize tensor operations - avoid unnecessary device transfers
        if label_embedding.device != self.device:
            label_embedding = label_embedding.to(self.device)
        label_embedding_np = label_embedding.cpu().numpy()

        # Create and cache UMAP model
        self._umap_model = UMAP(random_state=42, n_components=n_components)
        umap_features = self._umap_model.fit_transform(label_embedding_np)

        return torch.tensor(
            umap_features, device=self.device, dtype=label_embedding.dtype
        )

    def get_and_predict_umap(
        self, label_embedding, label_embeddings_new=None, n_components: int = 2
    ):
        self._ensure_cluster()

        # Optimize tensor operations
        if label_embedding.device != self.device:
            label_embedding = label_embedding.to(self.device)
        label_embedding_np = label_embedding.cpu().numpy()

        # Create and cache UMAP model
        self._umap_model = UMAP(random_state=42, n_components=n_components)
        umap_features = self._umap_model.fit_transform(label_embedding_np)

        # Predict UMAP features for new embeddings
        umap_features_new = None
        if label_embeddings_new is not None:
            if label_embeddings_new.device != self.device:
                label_embeddings_new = label_embeddings_new.to(self.device)
            label_embeddings_new_np = label_embeddings_new.cpu().numpy()
            umap_features_new = self._umap_model.transform(label_embeddings_new_np)
            umap_features_new = torch.tensor(
                umap_features_new, device=self.device, dtype=label_embeddings_new.dtype
            )

        umap_features = torch.tensor(
            umap_features, device=self.device, dtype=label_embedding.dtype
        )

        if umap_features_new is None:
            return umap_features, None
        return umap_features, umap_features_new

    def get_kmeans(self, umap_features, n_clusters):
        # Perform KMeans clustering on UMAP features
        self._ensure_cluster()

        umap_features_np = umap_features.cpu().numpy()

        # Cache and reuse KMeans model if same n_clusters
        if (
            self._kmeans_model is None
            or getattr(self._kmeans_model, "n_clusters", None) != n_clusters
        ):
            self._kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)

        self._kmeans_model.fit(umap_features_np)
        umap_labels = self._kmeans_model.labels_
        centers = self._kmeans_model.cluster_centers_

        umap_labels = torch.tensor(
            umap_labels, device=umap_features.device, dtype=torch.long
        )
        centers = torch.tensor(
            centers, device=umap_features.device, dtype=umap_features.dtype
        )

        return umap_labels, centers

    def get_hdbscan(
        self,
        umap_features,
        min_cluster_size=100,
        min_sample=50,
        method="leaf",
    ):  # leaf or eom
        self._ensure_cluster()

        umap_features_np = umap_features.cpu().numpy()
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_sample,
            cluster_selection_method=method,
        )
        hdbscan_model.fit(umap_features_np)
        umap_labels = hdbscan_model.labels_

        umap_labels = torch.tensor(
            umap_labels, device=umap_features.device, dtype=torch.long
        )

        return umap_labels, None

    def kmeans_update(
        self,
        umap_labels,
        original_embeddings,
        update_type="hard",
        alpha=0.1,
    ):
        device = original_embeddings.device
        num_clusters = umap_labels.max().item() + 1

        # Optimize cluster center calculation using scatter operations
        cluster_centers = torch.zeros(
            (num_clusters, original_embeddings.shape[1]),
            device=device,
            dtype=original_embeddings.dtype,
        )

        # Vectorized cluster center calculation
        print("Calculating cluster centers")
        cluster_counts = torch.bincount(umap_labels, minlength=num_clusters).float()

        # Use scatter_add for efficient cluster center computation
        expanded_labels = umap_labels.unsqueeze(1).expand(
            -1, original_embeddings.shape[1]
        )
        cluster_centers.scatter_add_(0, expanded_labels, original_embeddings)

        # Avoid division by zero
        cluster_counts = cluster_counts.clamp(min=1.0)
        cluster_centers = cluster_centers / cluster_counts.unsqueeze(1)

        print("Updating embeddings")
        if update_type == "hard":
            # Vectorized hard update using advanced indexing
            updated_embeddings = cluster_centers[umap_labels]
        else:
            # Soft update (if needed in the future)
            updated_embeddings = (
                1 - alpha
            ) * original_embeddings + alpha * cluster_centers[umap_labels]

        return updated_embeddings

    def hdbscan_update(
        self,
        umap_labels,
        original_embeddings,
        update_type="hard",
        alpha=0.5,
        update_noise="ignore",  # 'ignore' or 'assign'
        center_only=False,
    ):
        device = original_embeddings.device
        alpha = max(min(alpha, 0.99), 0.01)

        # Separate noise and non-noise points
        noise_mask = umap_labels == -1
        non_noise_mask = ~noise_mask
        unique_labels = umap_labels[non_noise_mask].unique()
        num_clusters = len(unique_labels)

        if num_clusters == 0:
            # Handle case with no valid clusters
            cluster_centers = torch.zeros(
                (1, original_embeddings.shape[-1]), device=device
            )
            return (
                original_embeddings,
                cluster_centers,
                torch.tensor([0], device=device),
            )

        cluster_centers = torch.zeros(
            (num_clusters, original_embeddings.shape[1]),
            device=device,
            dtype=original_embeddings.dtype,
        )

        # Create label mapping for efficient indexing
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}

        # Vectorized cluster center calculation for non-noise points
        print("Calculating cluster centers")
        cluster_counts = torch.zeros(num_clusters, device=device)

        for i, label in enumerate(unique_labels):
            mask = (umap_labels == label) & non_noise_mask
            if mask.sum() > 0:
                cluster_centers[i] = original_embeddings[mask].mean(dim=0)
                cluster_counts[i] = mask.sum().float()

        if center_only:
            return None, cluster_centers, cluster_counts

        updated_embeddings = original_embeddings.clone()

        # Handle noise assignment if requested
        if update_noise == "assign" and noise_mask.sum() > 0:
            print("Assigning noise points to nearest clusters")
            noise_embeddings = original_embeddings[noise_mask]

            # Compute distances and find nearest clusters
            distances = torch.cdist(noise_embeddings, cluster_centers)
            nearest_cluster_indices = distances.argmin(dim=1)
            nearest_cluster_labels = unique_labels[nearest_cluster_indices]

            # Update labels for noise points
            umap_labels = umap_labels.clone()  # Avoid in-place modification
            umap_labels[noise_mask] = nearest_cluster_labels
            noise_mask = torch.zeros_like(noise_mask)  # No more noise points

        # Vectorized embedding updates
        print("Updating embeddings")
        if update_type == "hard":
            # Create mapping tensor for all labels
            label_mapping = torch.full(
                (umap_labels.max().item() + 1,), -1, device=device, dtype=torch.long
            )
            for label, idx in label_to_idx.items():
                if label >= 0:  # Skip noise label
                    label_mapping[label] = idx

            # Update non-noise embeddings
            valid_mask = (umap_labels >= 0) & (umap_labels < len(label_mapping))
            if valid_mask.sum() > 0:
                valid_labels = umap_labels[valid_mask]
                center_indices = label_mapping[valid_labels]
                valid_center_mask = center_indices >= 0
                if valid_center_mask.sum() > 0:
                    valid_indices = torch.where(valid_mask)[0][valid_center_mask]
                    valid_center_indices = center_indices[valid_center_mask]
                    updated_embeddings[valid_indices] = cluster_centers[
                        valid_center_indices
                    ]

        elif update_type == "soft":
            # Similar vectorized approach for soft updates
            label_mapping = torch.full(
                (umap_labels.max().item() + 1,), -1, device=device, dtype=torch.long
            )
            for label, idx in label_to_idx.items():
                if label >= 0:
                    label_mapping[label] = idx

            valid_mask = (umap_labels >= 0) & (umap_labels < len(label_mapping))
            if valid_mask.sum() > 0:
                valid_labels = umap_labels[valid_mask]
                center_indices = label_mapping[valid_labels]
                valid_center_mask = center_indices >= 0
                if valid_center_mask.sum() > 0:
                    valid_indices = torch.where(valid_mask)[0][valid_center_mask]
                    valid_center_indices = center_indices[valid_center_mask]

                    updated_embeddings[valid_indices] = (
                        1 - alpha
                    ) * original_embeddings[valid_indices] + alpha * cluster_centers[
                        valid_center_indices
                    ]
        else:
            raise ValueError("update_type must be 'hard' or 'soft'.")

        return updated_embeddings, cluster_centers, cluster_counts


def test_kmeans():
    # Example usage of Clustering class
    random.seed(42)

    clustering = Clustering(device="cuda")

    label_embedding = torch.randn(1024, 512).to(clustering.device)  # Dummy data

    umap_features = clustering.get_umap(label_embedding)

    umap_labels, centers = clustering.get_kmeans(umap_features, n_clusters=1024 - 30)

    updated_embeddings = clustering.kmeans_update(
        umap_labels, label_embedding, update_type="hard", alpha=0.1
    )

    # Check if embeddings have been updated
    differences = torch.any(label_embedding != updated_embeddings, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)

    umap_features_new = clustering.get_umap(updated_embeddings)
    umap_labels_new, centers_new = clustering.get_kmeans(
        umap_features_new, n_clusters=1024 - 60
    )

    updated_embeddings_new = clustering.kmeans_update(
        umap_labels_new, updated_embeddings, update_type="hard", alpha=0.1
    )

    # Check if embeddings have been updated
    differences = torch.any(updated_embeddings != updated_embeddings_new, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)


def test_hdbscan():
    # Example usage of Clustering class
    random.seed(42)

    clustering = Clustering(device="cuda")

    # label_embedding = torch.randn(1024, 512).to(clustering.device)  # Dummy data

    num_clusters = 5
    num_points_per_cluster = 1000
    dim = 512

    clusters = []

    for k in range(num_clusters):
        # Generate a random mean vector for each cluster
        # Offset means to ensure clusters are well-separated
        mu_k = torch.randn(dim) * 10 + k * 100.0

        # Generate samples for the cluster0
        samples = torch.randn(num_points_per_cluster, dim) + mu_k
        clusters.append(samples)

    # Combine all clusters into one dataset
    label_embedding = torch.cat(clusters, dim=0).to(clustering.device)

    umap_features = clustering.get_umap(label_embedding)

    umap_labels, centers = clustering.get_hdbscan(umap_features)
    print(umap_labels.shape)

    print(type(umap_labels))

    updated_embeddings, centers, cluster_counts = clustering.hdbscan_update(
        umap_labels,
        label_embedding,
        update_type="hard",
        alpha=0.1,
        update_noise="ignore",
    )

    # Check if embeddings have been updated
    differences = torch.any(label_embedding != updated_embeddings, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)

    umap_features_new, _ = clustering.get_and_predict_umap(updated_embeddings)
    umap_labels_new, centers_new = clustering.get_hdbscan(umap_features_new)

    updated_embeddings_new, centers_new, cluster_counts_new = clustering.hdbscan_update(
        umap_labels_new,
        updated_embeddings,
        update_type="hard",
        alpha=0.1,
        update_noise="assign",
    )

    print(f"Centers: {centers_new.shape}")

    # Check if embeddings have been updated
    differences = torch.any(updated_embeddings != updated_embeddings_new, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)


def main():
    test_hdbscan()


if __name__ == "__main__":
    main()
