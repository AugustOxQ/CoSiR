import json
import os
import shutil
import tarfile
import time
import threading
import pickle
from datetime import datetime
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# ==================== Trainable Embedding Management ====================


class TrainableEmbeddingManager:
    """Manages per-sample trainable embeddings as PyTorch parameters"""

    def __init__(
        self,
        sample_ids: List[int],
        embedding_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        initialization_strategy: str = "normal",
    ):
        """
        Initialize trainable embeddings for samples

        Args:
            sample_ids: List of unique sample identifiers
            embedding_dim: Dimension of embedding vectors
            device: Device to store embeddings on
            initialization_strategy: How to initialize embeddings
        """
        self.sample_ids = sorted(sample_ids)
        self.embedding_dim = embedding_dim
        self.device = device

        # Create embedding lookup table as a single nn.Parameter
        init_tensor = self._initialize_embeddings(initialization_strategy).to(device)
        self.embeddings = nn.Parameter(init_tensor, requires_grad=True)

        # Fast sample_id -> index mapping
        self.id_to_index = {sid: idx for idx, sid in enumerate(self.sample_ids)}

        # Embedding versioning and checkpointing
        self.version = 0
        self.checkpoint_history = []

        # Training state
        self.frozen_indices = set()

    def _initialize_embeddings(self, strategy: str) -> torch.Tensor:
        """Initialize embeddings using various strategies"""
        n_samples = len(self.sample_ids)

        if strategy == "zeros":
            return torch.zeros(n_samples, self.embedding_dim)

        elif strategy == "normal":
            embeddings = torch.randn(n_samples, self.embedding_dim)
            embeddings.mul_(0.01)  # In-place operation to maintain leaf status
            return embeddings

        elif strategy == "xavier":
            embeddings = torch.empty(n_samples, self.embedding_dim)
            nn.init.xavier_uniform_(embeddings)
            return embeddings

        elif strategy == "kaiming":
            embeddings = torch.empty(n_samples, self.embedding_dim)
            nn.init.kaiming_uniform_(embeddings)
            return embeddings

        elif strategy == "uniform":
            return torch.rand(n_samples, self.embedding_dim) * 0.02 - 0.01

        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def get_embeddings(self, sample_ids: List[int]) -> torch.Tensor:
        """
        Get embeddings for specific samples

        Args:
            sample_ids: List of sample IDs to retrieve

        Returns:
            Tensor of shape (len(sample_ids), embedding_dim)
        """
        indices = [
            self.id_to_index[sid] for sid in sample_ids if sid in self.id_to_index
        ]
        if len(indices) != len(sample_ids):
            missing = [sid for sid in sample_ids if sid not in self.id_to_index]
            raise ValueError(f"Sample IDs not found: {missing}")
        return self.embeddings[indices]

    def get_embedding_by_index(self, indices: List[int]) -> torch.Tensor:
        """Get embeddings by direct indices (faster for batch operations)"""
        return self.embeddings[indices]

    def update_embeddings(self, sample_ids: List[int], new_embeddings: torch.Tensor):
        """
        Update specific embeddings (used during training)

        Args:
            sample_ids: Sample IDs to update
            new_embeddings: New embedding values
        """
        indices = [
            self.id_to_index[sid] for sid in sample_ids if sid in self.id_to_index
        ]
        with torch.no_grad():
            # Use .data to maintain leaf status and avoid breaking computational graph
            self.embeddings.data[indices] = new_embeddings.to(self.device)

    def get_all_embeddings(self) -> Tuple[List[int], torch.Tensor]:
        """Return all embeddings with their sample IDs"""
        return self.sample_ids.copy(), self.embeddings.clone()

    def get_similarity_matrix(
        self, sample_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix for embeddings

        Args:
            sample_ids: Subset of samples, or None for all samples

        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        if sample_ids is None:
            embeddings = self.embeddings
        else:
            embeddings = self.get_embeddings(sample_ids)

        # Cosine similarity
        normalized = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.mm(normalized, normalized.t())

        return similarity

    def get_nearest_neighbors(
        self, sample_id: int, k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors for a sample

        Args:
            sample_id: Query sample ID
            k: Number of neighbors to return

        Returns:
            List of (sample_id, similarity_score) tuples
        """
        query_embedding = self.get_embeddings([sample_id])

        # Compute similarities with all other embeddings
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(1), self.embeddings.unsqueeze(0), dim=2
        ).squeeze()

        # Get top-k (excluding self)
        _, indices = similarities.topk(k + 1)
        indices = indices[1:]  # Remove self

        neighbors = []
        for idx in indices:
            neighbor_id = self.sample_ids[idx.item()]
            similarity = similarities[idx].item()
            neighbors.append((neighbor_id, similarity))

        return neighbors

    def create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint for model saving"""
        checkpoint = {
            "embeddings": self.embeddings.detach().cpu(),
            "sample_ids": self.sample_ids,
            "embedding_dim": self.embedding_dim,
            "version": self.version,
            "timestamp": time.time(),
            "frozen_indices": list(self.frozen_indices),
            "device": self.device,
        }
        self.checkpoint_history.append(checkpoint["timestamp"])
        self.version += 1
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load from checkpoint"""
        self.embeddings.data = checkpoint["embeddings"].to(self.device)
        self.sample_ids = checkpoint["sample_ids"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.version = checkpoint.get("version", 0)
        self.frozen_indices = set(checkpoint.get("frozen_indices", []))

        # Rebuild index mapping
        self.id_to_index = {sid: idx for idx, sid in enumerate(self.sample_ids)}

    def get_optimizer_params(self) -> List[torch.nn.Parameter]:
        """Return parameters for optimizer"""
        return [self.embeddings]

    def freeze_embeddings(self, sample_ids: Optional[List[int]] = None):
        """
        Freeze specific embeddings or all embeddings

        Args:
            sample_ids: Sample IDs to freeze, or None to freeze all
        """
        if sample_ids is None:
            self.embeddings.requires_grad = False
            self.frozen_indices = set(range(len(self.sample_ids)))
        else:
            indices = [
                self.id_to_index[sid] for sid in sample_ids if sid in self.id_to_index
            ]
            self.frozen_indices.update(indices)

            # For partial freezing, we'd need custom autograd functions
            # This is a simplified version - full implementation would be more complex

    def unfreeze_embeddings(self, sample_ids: Optional[List[int]] = None):
        """Unfreeze specific embeddings or all embeddings"""
        if sample_ids is None:
            self.embeddings.requires_grad = True
            self.frozen_indices.clear()
        else:
            indices = [
                self.id_to_index[sid] for sid in sample_ids if sid in self.id_to_index
            ]
            self.frozen_indices.difference_update(indices)

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about current embeddings"""
        embeddings = self.embeddings.detach()

        return {
            "mean_norm": torch.norm(embeddings, p=2, dim=1).mean().item(),
            "std_norm": torch.norm(embeddings, p=2, dim=1).std().item(),
            "mean_values": embeddings.mean(dim=0).cpu(),
            "std_values": embeddings.std(dim=0).cpu(),
            "min_values": embeddings.min(dim=0)[0].cpu(),
            "max_values": embeddings.max(dim=0)[0].cpu(),
            "num_frozen": len(self.frozen_indices),
            "total_samples": len(self.sample_ids),
        }


class EmbeddingInitializer:
    """Advanced initialization strategies for embeddings"""

    @staticmethod
    def from_features(
        feature_manager,
        sample_ids: List[int],
        feature_type: str = "img_features",
        reduction: str = "identity",
    ) -> torch.Tensor:
        """
        Initialize embeddings from pre-extracted features

        Args:
            feature_manager: FeatureManager instance
            sample_ids: Sample IDs to initialize
            feature_type: Which feature type to use
            reduction: How to reduce feature dimensions if needed
        """
        features = feature_manager.get_features(sample_ids, [feature_type])
        feature_tensor = features[feature_type]

        if reduction == "identity":
            return feature_tensor.clone()
        elif reduction == "pca":
            # Implement PCA reduction
            pca = PCA(n_components=512)  # Or desired embedding dim
            reduced = pca.fit_transform(feature_tensor.cpu().numpy())
            return torch.from_numpy(reduced).float()
        elif reduction == "random_projection":
            # Johnson-Lindenstrauss random projection
            input_dim = feature_tensor.shape[1]
            target_dim = min(512, input_dim)  # Or desired embedding dim
            projection_matrix = torch.randn(input_dim, target_dim) / (input_dim**0.5)
            return torch.mm(feature_tensor, projection_matrix)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")

    @staticmethod
    def from_clustering(
        features: torch.Tensor, n_clusters: int, cluster_method: str = "kmeans"
    ) -> torch.Tensor:
        """
        Initialize embeddings as cluster centroids

        Args:
            features: Input features for clustering
            n_clusters: Number of clusters
            cluster_method: Clustering algorithm to use
        """

        if cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_centers = kmeans.fit(features.cpu().numpy()).cluster_centers_
            return torch.from_numpy(cluster_centers).float()

        elif cluster_method == "gaussian_mixture":
            gm = GaussianMixture(n_components=n_clusters, random_state=42)
            gm.fit(features.cpu().numpy())
            return torch.from_numpy(gm.means_).float()

        else:
            raise ValueError(f"Unknown clustering method: {cluster_method}")

    @staticmethod
    def from_pretrained_model(
        model, sample_ids: List[int], data_loader
    ) -> torch.Tensor:
        """Initialize embeddings from a pretrained model"""
        model.eval()
        embeddings_list = []

        with torch.no_grad():
            for batch in data_loader:
                # Assuming batch contains sample_ids and data
                batch_embeddings = model.encode(batch["data"])
                embeddings_list.append(batch_embeddings)

        return torch.cat(embeddings_list, dim=0)


class EmbeddingScheduler:
    """Manage embedding learning schedules and constraints"""

    def __init__(self, embedding_manager: TrainableEmbeddingManager):
        self.embedding_manager = embedding_manager
        self.learning_schedule = {}
        self.constraints = {}
        self.regularizers = {}

    def set_learning_rate_schedule(self, schedule: Dict[int, float]):
        """
        Set different learning rates for different epochs

        Args:
            schedule: Dict mapping epoch -> learning_rate
        """
        self.learning_schedule = schedule

    def add_similarity_constraint(
        self,
        sample_pairs: List[Tuple[int, int]],
        weight: float,
        target_similarity: float = 1.0,
    ):
        """
        Add similarity constraints between sample pairs

        Args:
            sample_pairs: List of (sample_id1, sample_id2) pairs
            weight: Constraint weight in loss function
            target_similarity: Target similarity value (1.0 for identical, 0.0 for orthogonal)
        """
        self.constraints["similarity"] = {
            "pairs": sample_pairs,
            "weight": weight,
            "target": target_similarity,
        }

    def add_diversity_constraint(
        self, sample_groups: List[List[int]], weight: float, min_distance: float = 0.5
    ):
        """
        Add diversity constraints within groups

        Args:
            sample_groups: List of sample ID groups that should be diverse
            weight: Constraint weight
            min_distance: Minimum distance between samples in group
        """
        self.constraints["diversity"] = {
            "groups": sample_groups,
            "weight": weight,
            "min_distance": min_distance,
        }

    def add_regularizer(self, regularizer_type: str, weight: float, **kwargs):
        """
        Add regularization terms

        Args:
            regularizer_type: Type of regularization ('l1', 'l2', 'orthogonal')
            weight: Regularization weight
        """
        self.regularizers[regularizer_type] = {"weight": weight, "params": kwargs}

    def get_constraint_loss(self) -> torch.Tensor:
        """Compute constraint losses for regularization"""
        total_loss = torch.tensor(0.0, device=self.embedding_manager.device)

        # Similarity constraints
        if "similarity" in self.constraints:
            constraint = self.constraints["similarity"]
            pairs = constraint["pairs"]
            weight = constraint["weight"]
            target = constraint["target"]

            for sid1, sid2 in pairs:
                try:
                    emb1 = self.embedding_manager.get_embeddings([sid1])
                    emb2 = self.embedding_manager.get_embeddings([sid2])

                    similarity = F.cosine_similarity(emb1, emb2, dim=1)
                    target_tensor = torch.tensor([target], device=similarity.device)

                    loss = F.mse_loss(similarity, target_tensor)
                    total_loss += weight * loss
                except ValueError:
                    # Skip if sample IDs not found
                    continue

        # Diversity constraints
        if "diversity" in self.constraints:
            constraint = self.constraints["diversity"]
            groups = constraint["groups"]
            weight = constraint["weight"]
            min_distance = constraint["min_distance"]

            for group in groups:
                if len(group) < 2:
                    continue

                try:
                    group_embeddings = self.embedding_manager.get_embeddings(group)

                    # Compute pairwise distances within group
                    distances = torch.cdist(group_embeddings, group_embeddings, p=2)

                    # Mask diagonal (self-distances)
                    mask = ~torch.eye(len(group), dtype=bool, device=distances.device)
                    group_distances = distances[mask]

                    # Penalize distances smaller than min_distance
                    violation = F.relu(min_distance - group_distances)
                    diversity_loss = violation.mean()

                    total_loss += weight * diversity_loss
                except ValueError:
                    # Skip if sample IDs not found
                    continue

        return total_loss

    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization losses"""
        total_loss = torch.tensor(0.0, device=self.embedding_manager.device)
        embeddings = self.embedding_manager.embeddings

        # L1 regularization
        if "l1" in self.regularizers:
            weight = self.regularizers["l1"]["weight"]
            l1_loss = torch.norm(embeddings, p=1)
            total_loss += weight * l1_loss

        # L2 regularization
        if "l2" in self.regularizers:
            weight = self.regularizers["l2"]["weight"]
            l2_loss = torch.norm(embeddings, p=2)
            total_loss += weight * l2_loss

        # Orthogonality regularization
        if "orthogonal" in self.regularizers:
            weight = self.regularizers["orthogonal"]["weight"]

            # Encourage embeddings to be orthogonal
            normalized = F.normalize(embeddings, p=2, dim=1)
            gram_matrix = torch.mm(normalized, normalized.t())

            # Penalize off-diagonal elements
            identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            orthogonal_loss = torch.norm(gram_matrix - identity, p="fro")

            total_loss += weight * orthogonal_loss

        return total_loss

    def get_total_loss(self) -> torch.Tensor:
        """Get total loss from constraints and regularization"""
        constraint_loss = self.get_constraint_loss()
        regularization_loss = self.get_regularization_loss()
        return constraint_loss + regularization_loss


# ==================== Experiment Management System ====================


class ExperimentPaths:
    """Centralized path management for experiments"""

    def __init__(self, base_dir: Path):
        self.base = base_dir

        # Standard paths
        self.checkpoints = base_dir / "checkpoints"
        self.logs = base_dir / "logs"
        self.plots = base_dir / "plots"
        self.embeddings = base_dir / "embeddings"
        self.features = base_dir / "features"
        self.results = base_dir / "results"
        self.artifacts = base_dir / "artifacts"
        self.configs = base_dir / "configs"
        self.scripts = base_dir / "scripts"

    def get_epoch_dir(self, epoch: int) -> Path:
        """Get epoch-specific directory"""
        epoch_dir = self.base / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        return epoch_dir

    def get_plot_path(self, plot_name: str, epoch: Optional[int] = None) -> Path:
        """Get standardized plot path"""
        if epoch is not None:
            return self.plots / f"{plot_name}_epoch_{epoch}.png"
        else:
            return self.plots / f"{plot_name}.png"

    def get_checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        """Get checkpoint path"""
        if epoch is None:
            return self.checkpoints / "latest.pt"
        else:
            return self.checkpoints / f"checkpoint_epoch_{epoch}.pt"


class ExperimentContext:
    """Context manager for experiment lifecycle"""

    def __init__(
        self,
        name: str,
        directory: Path,
        config: Dict[str, Any],
        tags: List[str],
        description: Optional[str] = None,
        parent_experiment: Optional[str] = None,
    ):

        self.name = name
        self.directory = directory
        self.config = config
        self.tags = tags
        self.description = description or ""
        self.parent_experiment = parent_experiment

        # Experiment state
        self.status = "created"
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self.current_epoch = 0

        # Metrics and artifacts
        self.metrics_history = []
        self.artifacts = {}
        self.notes = []

        # Paths
        self.paths = ExperimentPaths(directory)

        # Create directories first
        self._create_directories()

        # Save initial config
        self._save_config()

    def __enter__(self):
        self.status = "running"
        self.start_time = time.time()
        self._save_status()

        # Create run-specific log
        run_log = {"started": self.start_time, "config": self.config, "tags": self.tags}

        with open(self.paths.logs / "run_log.json", "w") as f:
            json.dump(run_log, f, indent=2, default=str)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

        if exc_type is None:
            self.status = "completed"
        else:
            self.status = "failed"
            # Log error information
            self._log_error(exc_type, exc_val, exc_tb)

        self._save_status()
        self._save_final_summary()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.paths.checkpoints,
            self.paths.logs,
            self.paths.plots,
            self.paths.embeddings,
            self.paths.features,
            self.paths.results,
            self.paths.artifacts,
            self.paths.configs,
            self.paths.scripts,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_metadata(
        cls, directory: Path, metadata: Dict[str, Any]
    ) -> "ExperimentContext":
        """Create ExperimentContext from saved metadata"""
        context = cls(
            name=metadata["name"],
            directory=directory,
            config=metadata.get("config", {}),
            tags=metadata.get("tags", []),
            description=metadata.get("description", ""),
            parent_experiment=metadata.get("parent_experiment"),
        )

        context.status = metadata.get("status", "unknown")
        context.created_time = metadata.get("created", time.time())

        # Load existing data
        context._load_existing_data()

        return context

    def _save_config(self):
        """Save current configuration"""
        config_file = self.paths.configs / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _save_status(self):
        """Save current status"""
        status_info = {
            "status": self.status,
            "created": self.created_time,
            "started": self.start_time,
            "ended": self.end_time,
            "current_epoch": self.current_epoch,
            "last_updated": time.time(),
        }

        status_file = self.directory / "status.json"
        with open(status_file, "w") as f:
            json.dump(status_info, f, indent=2, default=str)

    def _log_error(self, exc_type, exc_val, exc_tb):
        """Log error information"""
        import traceback

        error_info = {
            "timestamp": time.time(),
            "error_type": exc_type.__name__ if exc_type else None,
            "error_message": str(exc_val) if exc_val else None,
            "traceback": traceback.format_tb(exc_tb) if exc_tb else None,
        }

        error_file = self.paths.logs / "error.json"
        with open(error_file, "w") as f:
            json.dump(error_info, f, indent=2, default=str)

    def _save_final_summary(self):
        """Save final experiment summary"""
        summary = {
            "name": self.name,
            "status": self.status,
            "created": self.created_time,
            "started": self.start_time,
            "ended": self.end_time,
            "duration": (
                self.end_time - self.start_time
                if self.start_time and self.end_time
                else None
            ),
            "total_epochs": self.current_epoch,
            "final_metrics": self.get_final_metrics(),
            "config": self.config,
            "tags": self.tags,
            "description": self.description,
            "artifacts": list(self.artifacts.keys()),
            "notes": self.notes,
        }

        summary_file = self.directory / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _load_existing_data(self):
        """Load existing data from previous runs"""
        # Load metrics history
        if self.paths.logs.exists():
            metrics_files = sorted(self.paths.logs.glob("metrics_epoch_*.json"))
            for metrics_file in metrics_files:
                try:
                    with open(metrics_file) as f:
                        metric_data = json.load(f)
                        self.metrics_history.append(metric_data)
                except:
                    continue

        # Update current epoch
        if self.metrics_history:
            self.current_epoch = max(m.get("epoch", 0) for m in self.metrics_history)

        # Load artifacts info
        artifacts_file = self.directory / "artifacts.json"
        if artifacts_file.exists():
            with open(artifacts_file) as f:
                self.artifacts = json.load(f)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Log metrics for current epoch/step

        Args:
            metrics: Dict of metric_name -> value
            epoch: Epoch number (uses current_epoch if None)
            step: Step number within epoch
        """
        if epoch is None:
            epoch = self.current_epoch

        metric_entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics.copy(),
        }

        self.metrics_history.append(metric_entry)

        # Save to file
        if step is None:
            metrics_file = self.paths.logs / f"metrics_epoch_{epoch}.json"
        else:
            metrics_file = self.paths.logs / f"metrics_epoch_{epoch}_step_{step}.json"

        with open(metrics_file, "w") as f:
            json.dump(metric_entry, f, indent=2, default=str)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        embedding_manager: TrainableEmbeddingManager,
        optimizer_state: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Save comprehensive checkpoint

        Args:
            model_state: Model state dict
            embedding_manager: Embedding manager instance
            optimizer_state: Optimizer state dict
            epoch: Epoch number
            additional_data: Any additional data to save
        """
        if epoch is None:
            epoch = self.current_epoch

        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "embeddings_checkpoint": embedding_manager.create_checkpoint(),
            "optimizer_state": optimizer_state,
            "config": self.config,
            "timestamp": time.time(),
            "experiment_name": self.name,
        }

        if additional_data:
            checkpoint["additional_data"] = additional_data

        checkpoint_path = self.paths.checkpoints / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Keep link to latest checkpoint
        latest_path = self.paths.checkpoints / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()

        # Create relative symlink
        try:
            latest_path.symlink_to(checkpoint_path.name)
        except OSError:
            # Fallback for systems that don't support symlinks
            shutil.copy2(checkpoint_path, latest_path)

        # Keep only last N checkpoints to save space
        self._cleanup_old_checkpoints(keep_last=5)

    def load_checkpoint(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        Load checkpoint

        Args:
            epoch: Specific epoch to load, or None for latest
        """
        if epoch is None:
            checkpoint_path = self.paths.checkpoints / "latest.pt"
        else:
            checkpoint_path = self.paths.checkpoints / f"checkpoint_epoch_{epoch}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location="cpu")

    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Keep only the last N checkpoints"""
        checkpoint_files = []
        for f in self.paths.checkpoints.glob("checkpoint_epoch_*.pt"):
            try:
                epoch_num = int(f.stem.split("_")[-1])
                checkpoint_files.append((epoch_num, f))
            except ValueError:
                continue

        checkpoint_files.sort(key=lambda x: x[0])  # Sort by epoch number

        if len(checkpoint_files) > keep_last:
            for _, old_checkpoint in checkpoint_files[:-keep_last]:
                try:
                    old_checkpoint.unlink()
                except OSError:
                    pass

    def save_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = "pickle",
        description: Optional[str] = None,
    ):
        """
        Save experiment artifact

        Args:
            name: Artifact name
            data: Data to save
            artifact_type: Type of artifact ('pickle', 'json', 'torch', 'numpy')
            description: Human-readable description
        """
        if artifact_type == "pickle":
            artifact_path = self.paths.artifacts / f"{name}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(data, f)

        elif artifact_type == "json":
            artifact_path = self.paths.artifacts / f"{name}.json"
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif artifact_type == "torch":
            artifact_path = self.paths.artifacts / f"{name}.pt"
            torch.save(data, artifact_path)

        elif artifact_type == "numpy":
            artifact_path = self.paths.artifacts / f"{name}.npy"
            np.save(artifact_path, data)

        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

        # Register artifact
        self.artifacts[name] = {
            "path": str(artifact_path),
            "type": artifact_type,
            "description": description or "",
            "created": time.time(),
            "size_bytes": artifact_path.stat().st_size,
        }

        # Save artifacts registry
        artifacts_file = self.directory / "artifacts.json"
        with open(artifacts_file, "w") as f:
            json.dump(self.artifacts, f, indent=2, default=str)

    def load_artifact(self, name: str) -> Any:
        """Load experiment artifact"""
        if name not in self.artifacts:
            raise ValueError(f"Artifact {name} not found")

        artifact_info = self.artifacts[name]
        artifact_path = Path(artifact_info["path"])
        artifact_type = artifact_info["type"]

        if artifact_type == "pickle":
            with open(artifact_path, "rb") as f:
                return pickle.load(f)

        elif artifact_type == "json":
            with open(artifact_path) as f:
                return json.load(f)

        elif artifact_type == "torch":
            return torch.load(artifact_path, map_location="cpu")

        elif artifact_type == "numpy":
            return np.load(artifact_path)

        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

    def add_note(self, note: str, category: str = "general"):
        """Add a note to the experiment"""
        note_entry = {"timestamp": time.time(), "category": category, "note": note}

        self.notes.append(note_entry)

        # Save to file
        notes_file = self.paths.logs / "notes.json"
        with open(notes_file, "w") as f:
            json.dump(self.notes, f, indent=2, default=str)

    def get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics from the experiment"""
        if not self.metrics_history:
            return {}

        # Return metrics from the last epoch
        final_entry = max(self.metrics_history, key=lambda x: x.get("epoch", 0))
        return final_entry.get("metrics", {})

    def get_metrics_dataframe(self):
        """Get metrics as pandas DataFrame for analysis"""

        records = []
        for entry in self.metrics_history:
            record = {
                "epoch": entry.get("epoch", 0),
                "step": entry.get("step"),
                "timestamp": entry.get("timestamp", 0),
            }
            record.update(entry.get("metrics", {}))
            records.append(record)

        import pandas as pd

        return pd.DataFrame(records)

    def plot_metrics(
        self, metric_names: Optional[List[str]] = None, save_path: Optional[Path] = None
    ):
        """Plot experiment metrics"""

        df = self.get_metrics_dataframe()
        if isinstance(df, list) or (hasattr(df, "empty") and df.empty):
            print("No metrics to plot")
            return

        if metric_names is None:
            # Plot all numeric columns except epoch, step, timestamp
            import pandas as pd

            metric_names = [
                col
                for col in df.columns
                if col not in ["epoch", "step", "timestamp"]
                and pd.api.types.is_numeric_dtype(df[col])
            ]

        fig, axes = plt.subplots(
            len(metric_names), 1, figsize=(10, 3 * len(metric_names))
        )
        if len(metric_names) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_names):
            if metric in df.columns:
                ax.plot(df["epoch"], df[metric], marker="o")
                ax.set_title(f"{metric} vs Epoch")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = self.paths.get_plot_path("metrics_history")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


class ExperimentManager:
    """Comprehensive experiment lifecycle management"""

    def __init__(self, base_experiments_dir: Union[str, Path] = "experiments"):
        self.base_dir = Path(base_experiments_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.current_experiment = None
        self.experiment_registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry"""
        registry_file = self.base_dir / "experiment_registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                return json.load(f)
        return {"experiments": {}, "tags": defaultdict(list)}

    def _save_registry(self):
        """Save experiment registry"""
        registry_file = self.base_dir / "experiment_registry.json"
        with open(registry_file, "w") as f:
            json.dump(self.experiment_registry, f, indent=2, default=str)

    def create_experiment(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        parent_experiment: Optional[str] = None,
    ) -> ExperimentContext:
        """
        Create new experiment with automatic naming and structure

        Args:
            name: Experiment name (auto-generated if None)
            config: Experiment configuration dict
            tags: List of tags for categorization
            description: Human-readable description
            parent_experiment: Parent experiment name for hierarchical experiments
        """
        # Generate experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name is None:
            name = f"exp_{timestamp}"
        else:
            name = f"{timestamp}_{name.replace(' ', '_')}"

        # Ensure unique name
        counter = 1
        original_name = name
        while (self.base_dir / name).exists():
            name = f"{original_name}_{counter}"
            counter += 1

        # Create experiment directory
        exp_dir = self.base_dir / name
        exp_dir.mkdir(exist_ok=True)

        # Create experiment context
        experiment = ExperimentContext(
            name=name,
            directory=exp_dir,
            config=config or {},
            tags=tags or [],
            description=description,
            parent_experiment=parent_experiment,
        )

        # Initialize standard directories
        self._create_standard_structure(experiment)

        # Save metadata
        self._save_experiment_metadata(experiment)

        # Update registry
        self._register_experiment(experiment)

        self.current_experiment = experiment
        return experiment

    def load_experiment(self, experiment_name: str) -> ExperimentContext:
        """
        Load existing experiment

        Args:
            experiment_name: Name of experiment to load
        """
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            # Try to find by partial name
            matches = [
                d
                for d in self.base_dir.iterdir()
                if d.is_dir() and experiment_name in d.name
            ]
            if len(matches) == 1:
                exp_dir = matches[0]
                experiment_name = exp_dir.name
            else:
                raise ValueError(
                    f"Experiment {experiment_name} not found. "
                    f"Matches: {[m.name for m in matches]}"
                )

        metadata = self._load_experiment_metadata(exp_dir)
        experiment = ExperimentContext.from_metadata(exp_dir, metadata)

        self.current_experiment = experiment
        return experiment

    def _create_standard_structure(self, experiment: ExperimentContext):
        """Create standard directory structure"""
        directories = [
            "checkpoints",
            "logs",
            "plots",
            "embeddings",
            "features",
            "results",
            "artifacts",
            "configs",
            "scripts",
        ]

        for dir_name in directories:
            (experiment.directory / dir_name).mkdir(exist_ok=True)

    def _save_experiment_metadata(self, experiment: ExperimentContext):
        """Save experiment metadata"""
        metadata = {
            "name": experiment.name,
            "created": experiment.created_time,
            "config": experiment.config,
            "tags": experiment.tags,
            "description": experiment.description,
            "parent_experiment": experiment.parent_experiment,
            "status": experiment.status,
            "directory": str(experiment.directory),
        }

        metadata_file = experiment.directory / "experiment_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_experiment_metadata(self, exp_dir: Path) -> Dict[str, Any]:
        """Load experiment metadata"""
        metadata_file = exp_dir / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        else:
            # Fallback for legacy experiments
            return {
                "name": exp_dir.name,
                "created": time.time(),
                "config": {},
                "tags": [],
                "description": "",
                "parent_experiment": None,
                "status": "unknown",
            }

    def _register_experiment(self, experiment: ExperimentContext):
        """Register experiment in global registry"""
        exp_info = {
            "created": experiment.created_time,
            "status": experiment.status,
            "tags": experiment.tags,
            "description": experiment.description,
            "parent_experiment": experiment.parent_experiment,
        }

        self.experiment_registry["experiments"][experiment.name] = exp_info

        # Update tag registry
        for tag in experiment.tags:
            if experiment.name not in self.experiment_registry["tags"][tag]:
                self.experiment_registry["tags"][tag].append(experiment.name)

        self._save_registry()

    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        parent: Optional[str] = None,
        sort_by: str = "created",
    ) -> List[Dict[str, Any]]:
        """
        List experiments with filtering options

        Args:
            tags: Filter by tags (OR operation)
            status: Filter by status
            parent: Filter by parent experiment
            sort_by: Sort key ('created', 'name', 'status')
        """
        experiments = []

        for exp_name, exp_info in self.experiment_registry["experiments"].items():
            # Apply filters
            if tags and not any(tag in exp_info.get("tags", []) for tag in tags):
                continue
            if status and exp_info.get("status") != status:
                continue
            if parent and exp_info.get("parent_experiment") != parent:
                continue

            experiments.append(
                {
                    "name": exp_name,
                    "created": exp_info.get("created"),
                    "status": exp_info.get("status", "unknown"),
                    "tags": exp_info.get("tags", []),
                    "description": exp_info.get("description", ""),
                    "parent_experiment": exp_info.get("parent_experiment"),
                }
            )

        # Sort experiments
        if sort_by == "created":
            experiments.sort(key=lambda x: x["created"], reverse=True)
        elif sort_by == "name":
            experiments.sort(key=lambda x: x["name"])
        elif sort_by == "status":
            experiments.sort(key=lambda x: x["status"])

        return experiments

    def archive_experiment(
        self,
        experiment_name: str,
        archive_path: Optional[Path] = None,
        remove_original: bool = True,
    ):
        """
        Archive completed experiment

        Args:
            experiment_name: Name of experiment to archive
            archive_path: Custom archive location
            remove_original: Whether to remove original after archiving
        """
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_name} not found")

        if archive_path is None:
            archive_dir = self.base_dir / "archived"
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / f"{experiment_name}.tar.gz"

        # Create archive
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(exp_dir, arcname=experiment_name)

        print(f"Archived experiment to {archive_path}")

        # Update registry
        if experiment_name in self.experiment_registry["experiments"]:
            self.experiment_registry["experiments"][experiment_name][
                "status"
            ] = "archived"
            self.experiment_registry["experiments"][experiment_name]["archive_path"] = (
                str(archive_path)
            )
            self._save_registry()

        # Remove original if requested
        if remove_original:
            shutil.rmtree(exp_dir)
            print(f"Removed original experiment directory")

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        comparison = {
            "experiments": {},
            "common_config": {},
            "config_differences": {},
            "metric_comparison": {},
        }

        for exp_name in experiment_names:
            try:
                exp = self.load_experiment(exp_name)
                comparison["experiments"][exp_name] = {
                    "config": exp.config,
                    "status": exp.status,
                    "created": exp.created_time,
                    "final_metrics": exp.get_final_metrics(),
                }
            except Exception as e:
                comparison["experiments"][exp_name] = {"error": str(e)}

        # Find common configuration
        if len(comparison["experiments"]) > 1:
            configs = [
                exp_data.get("config", {})
                for exp_data in comparison["experiments"].values()
                if "config" in exp_data
            ]
            if configs:
                common_keys = set(configs[0].keys())
                for config in configs[1:]:
                    common_keys &= set(config.keys())

                for key in common_keys:
                    values = [config[key] for config in configs]
                    if all(v == values[0] for v in values):
                        comparison["common_config"][key] = values[0]
                    else:
                        comparison["config_differences"][key] = {
                            exp_name: config.get(key)
                            for exp_name, config in zip(
                                comparison["experiments"].keys(), configs
                            )
                        }

        return comparison

    def cleanup_failed_experiments(self, older_than_days: int = 7):
        """Clean up failed experiments older than specified days"""
        cutoff_time = time.time() - (older_than_days * 24 * 3600)

        to_remove = []
        for exp_name, exp_info in self.experiment_registry["experiments"].items():
            if (
                exp_info.get("status") == "failed"
                and exp_info.get("created", 0) < cutoff_time
            ):
                to_remove.append(exp_name)

        for exp_name in to_remove:
            exp_dir = self.base_dir / exp_name
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
                print(f"Cleaned up failed experiment: {exp_name}")

            # Remove from registry
            if exp_name in self.experiment_registry["experiments"]:
                del self.experiment_registry["experiments"][exp_name]

        if to_remove:
            self._save_registry()


# ==================== Testing and Example Usage ====================


def test_trainable_embedding_manager():
    """Test TrainableEmbeddingManager functionality"""
    print("Testing TrainableEmbeddingManager...")

    # Test basic functionality
    sample_ids = [10, 20, 30, 40, 50]
    embedding_dim = 128

    # Test initialization
    manager = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=embedding_dim,
        initialization_strategy="normal",
    )

    print(f"Created embeddings: {manager.embeddings.shape}")
    print(f"Sample ID mapping: {manager.id_to_index}")

    # Test getting embeddings
    embeddings = manager.get_embeddings([10, 30, 50])
    print(f"Retrieved embeddings shape: {embeddings.shape}")

    # Test similarity computation
    similarity_matrix = manager.get_similarity_matrix([10, 20, 30])
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Test nearest neighbors
    neighbors = manager.get_nearest_neighbors(10, k=2)
    print(f"Nearest neighbors for sample 10: {neighbors}")

    # Test statistics
    stats = manager.get_embedding_statistics()
    print(f"Embedding statistics: mean_norm={stats['mean_norm']:.4f}")

    # Test checkpointing
    checkpoint = manager.create_checkpoint()
    print(f"Created checkpoint version: {checkpoint['version']}")

    # Test updating
    new_embeddings = torch.randn(2, embedding_dim)
    manager.update_embeddings([10, 20], new_embeddings)
    print("Updated embeddings successfully")

    print("TrainableEmbeddingManager tests passed!\n")


def test_embedding_scheduler():
    """Test EmbeddingScheduler functionality"""
    print("Testing EmbeddingScheduler...")

    sample_ids = [1, 2, 3, 4, 5]
    manager = TrainableEmbeddingManager(sample_ids, 64)
    scheduler = EmbeddingScheduler(manager)

    # Add constraints and regularizers
    scheduler.add_similarity_constraint([(1, 2), (3, 4)], weight=1.0)
    scheduler.add_diversity_constraint([[1, 2, 3]], weight=0.5)
    scheduler.add_regularizer("l2", weight=0.01)
    scheduler.add_regularizer("orthogonal", weight=0.001)

    # Test loss computation
    constraint_loss = scheduler.get_constraint_loss()
    regularization_loss = scheduler.get_regularization_loss()
    total_loss = scheduler.get_total_loss()

    print(f"Constraint loss: {constraint_loss.item():.6f}")
    print(f"Regularization loss: {regularization_loss.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")

    print("EmbeddingScheduler tests passed!\n")


def test_experiment_manager():
    """Test ExperimentManager functionality"""
    print("Testing ExperimentManager...")

    # Create experiment manager
    exp_manager = ExperimentManager(
        base_experiments_dir="/project/Deep-Clustering/res/test_experiments"
    )

    # Test experiment creation
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "embedding_dim": 256,
        "epochs": 10,
    }

    with exp_manager.create_experiment(
        name="test_deep_clustering",
        config=config,
        tags=["clustering", "test"],
        description="Test experiment for deep clustering",
    ) as experiment:

        print(f"Created experiment: {experiment.name}")
        print(f"Experiment directory: {experiment.directory}")

        # Test metrics logging
        for epoch in range(3):
            experiment.current_epoch = epoch
            experiment.log_metrics(
                {"train_loss": 1.0 - 0.1 * epoch, "val_accuracy": 0.5 + 0.1 * epoch},
                epoch,
            )

        # Test artifact saving
        test_data = {"test_key": "test_value", "numbers": [1, 2, 3]}
        experiment.save_artifact("test_data", test_data, "json", "Test artifact")

        # Test embedding manager integration
        sample_ids = [1, 2, 3, 4, 5]
        embedding_manager = TrainableEmbeddingManager(
            sample_ids, config["embedding_dim"]
        )

        # Test checkpoint saving
        experiment.save_checkpoint(
            model_state={"dummy": "model_state"},
            embedding_manager=embedding_manager,
            optimizer_state={"dummy": "optimizer_state"},
            epoch=2,
        )

        # Test note adding
        experiment.add_note("Test completed successfully", "completion")

        print(f"Logged {len(experiment.metrics_history)} metric entries")
        print(f"Saved {len(experiment.artifacts)} artifacts")
        print(f"Added {len(experiment.notes)} notes")

    print(f"Experiment completed with status: {experiment.status}")

    # Test experiment loading
    loaded_experiment = exp_manager.load_experiment(experiment.name)
    print(f"Loaded experiment: {loaded_experiment.name}")
    print(f"Final metrics: {loaded_experiment.get_final_metrics()}")

    # Test experiment listing
    experiments = exp_manager.list_experiments(tags=["test"])
    print(f"Found {len(experiments)} test experiments")

    print("ExperimentManager tests passed!\n")


def test_integrated_system():
    """Test integrated feature management and experiment system"""
    print("Testing integrated system...")

    # Import the feature management from the other file
    import sys

    sys.path.append("/project/Deep-Clustering/.improvement/tmpfile")
    from feature_management_tmp import FeatureManager

    # Create managers
    feature_manager = FeatureManager(
        "/project/Deep-Clustering/res/test_features_integrated", chunk_size=10
    )
    exp_manager = ExperimentManager(
        "/project/Deep-Clustering/res/test_experiments_integrated"
    )

    # Create test data
    sample_ids = list(range(20))
    img_features = torch.randn(20, 512)
    txt_features = torch.randn(20, 512)
    txt_full = ["sample text"] * 20

    # Add features in chunks
    for i in range(0, 20, 10):
        chunk_sample_ids = sample_ids[i : i + 10]
        chunk_img = img_features[i : i + 10]
        chunk_txt = txt_features[i : i + 10]
        chunk_txt_full = txt_full[i : i + 10]

        feature_manager.add_features_chunk(
            i // 10, chunk_img, chunk_txt, chunk_txt_full, chunk_sample_ids
        )

    # Create experiment with integrated workflow
    config = {
        "feature_manager": "integrated",
        "embedding_dim": 256,
        "learning_rate": 0.001,
    }

    with exp_manager.create_experiment(
        name="integrated_test", config=config, tags=["integration", "test"]
    ) as experiment:

        # Initialize embeddings from features using EmbeddingInitializer
        embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids, embedding_dim=config["embedding_dim"]
        )

        # Test loading features
        test_ids = sample_ids[:5]
        loaded_features = feature_manager.get_features(test_ids)
        print(f"Loaded features for integration test: {list(loaded_features.keys())}")

        # Test embedding operations
        embeddings = embedding_manager.get_embeddings(test_ids)
        print(f"Retrieved embeddings shape: {embeddings.shape}")

        # Simulate training loop
        for epoch in range(3):
            experiment.current_epoch = epoch

            # Mock training metrics
            train_loss = 1.0 - 0.1 * epoch
            embedding_stats = embedding_manager.get_embedding_statistics()

            experiment.log_metrics(
                {
                    "train_loss": train_loss,
                    "embedding_mean_norm": embedding_stats["mean_norm"],
                    "cache_hits": feature_manager.cache.stats["hits"],
                    "cache_misses": feature_manager.cache.stats["misses"],
                },
                epoch,
            )

        # Save final artifacts
        final_embeddings = embedding_manager.get_all_embeddings()
        experiment.save_artifact("final_embeddings", final_embeddings, "torch")
        experiment.save_artifact(
            "feature_cache_stats", feature_manager.cache.stats, "json"
        )

        print("Integrated system test completed successfully")

    print(f"Integrated experiment status: {experiment.status}")
    print("Integrated system tests passed!\n")


def main():
    """Run all tests"""
    print("=== Testing Embedding and Experiment Management System ===\n")

    # Test individual components
    test_trainable_embedding_manager()
    test_embedding_scheduler()
    test_experiment_manager()

    # Test integrated system
    test_integrated_system()

    print("=== All tests completed successfully! ===")


if __name__ == "__main__":
    main()
