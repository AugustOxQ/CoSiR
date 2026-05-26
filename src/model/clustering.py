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

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UMAP_vis:
    def __init__(self, device="cuda"):
        self.device = device
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

    def learn_umap(self, embedding, n_components: int = 2, close_cluster: bool = True):
        # Perform UMAP dimensionality reduction on embeddings
        self._ensure_cluster()

        if isinstance(embedding, torch.Tensor):
            label_embedding_np = embedding.cpu().numpy()
        else:
            label_embedding_np = embedding

        # Create and cache UMAP model
        self.local_model = UMAP(random_state=42, n_components=n_components)
        umap_features = self.local_model.fit_transform(label_embedding_np)

        if close_cluster:
            self.close_cluster()

        return umap_features

    def predict_umap(self, new_embedding):
        self._ensure_cluster()
        if self.local_model is None:
            raise ValueError("UMAP model has not been trained yet.")

        # Handle both numpy arrays and torch tensors
        if isinstance(new_embedding, torch.Tensor):
            new_embedding_np = new_embedding.cpu().numpy()
        else:
            new_embedding_np = new_embedding

        umap_features_new = self.local_model.transform(new_embedding_np)

        return umap_features_new


class Clustering:
    def __init__(self, device="cuda"):
        self.device = device
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
        if isinstance(label_embedding, torch.Tensor):
            label_embedding_np = label_embedding.cpu().numpy()

        # Create and cache UMAP model
        self._umap_model = UMAP(random_state=42, n_components=n_components)
        umap_features = self._umap_model.fit_transform(label_embedding_np)

        return umap_features

    def get_and_predict_umap(
        self, label_embedding, label_embeddings_new=None, n_components: int = 2
    ):
        self._ensure_cluster()

        if isinstance(label_embedding, torch.Tensor):
            label_embedding_np = label_embedding.cpu().numpy()

        # Create and cache UMAP model
        self._umap_model = UMAP(random_state=42, n_components=n_components)
        umap_features = self._umap_model.fit_transform(label_embedding_np)

        # Predict UMAP features for new embeddings
        umap_features_new = None
        if label_embeddings_new is not None:
            label_embeddings_new_np = label_embeddings_new.cpu().numpy()
            umap_features_new = self._umap_model.transform(label_embeddings_new_np)

        return umap_features, umap_features_new

    def get_kmeans(self, umap_features, n_clusters):
        # Perform KMeans clustering on UMAP features
        self._ensure_cluster()
        if isinstance(umap_features, torch.Tensor):
            umap_features_np = umap_features.cpu().numpy()
        else:
            umap_features_np = umap_features

        # Cache and reuse KMeans model if same n_clusters
        if (
            self._kmeans_model is None
            or getattr(self._kmeans_model, "n_clusters", None) != n_clusters
        ):
            self._kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)

        self._kmeans_model.fit(umap_features_np)
        umap_labels = self._kmeans_model.labels_
        centers = self._kmeans_model.cluster_centers_

        return umap_labels, centers

    def get_hdbscan(
        self,
        umap_features,
        min_cluster_size=100,
        min_sample=50,
        method="leaf",
    ):  # leaf or eom
        # Note: cuml.cluster.HDBSCAN is single-GPU and does not use the Dask cluster,
        # so _ensure_cluster / close_cluster are not called here.
        start_time = time.time()
        if isinstance(umap_features, torch.Tensor):
            umap_features_np = umap_features.cpu().numpy()
        else:
            umap_features_np = umap_features

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_sample,
            cluster_selection_method=method,
        )
        hdbscan_model.fit(umap_features_np)
        umap_labels = hdbscan_model.labels_
        del hdbscan_model
        torch.cuda.empty_cache()

        end_time = time.time()
        print(f"Time taken to get HDBSCAN labels: {end_time - start_time} seconds")

        return umap_labels, None

