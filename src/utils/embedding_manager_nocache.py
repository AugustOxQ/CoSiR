"""
Simplified TrainableEmbeddingManager without caching system
Created: 2025-01-13

This is a clean, minimal implementation that:
- Removes all caching complexity
- Keeps only essential functions
- Direct disk I/O for simplicity and reliability
- Easy to understand and maintain
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


class TrainableEmbeddingManager:
    """Manages per-sample trainable embeddings with simple disk storage"""

    def __init__(
        self,
        sample_ids: List[int],
        embedding_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        initialization_strategy: str = "normal",
        storage_mode: str = "disk",
        embeddings_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        feature_manager: Optional[object] = None,
        auto_sync: bool = True,
        # Deprecated parameters (kept for compatibility, ignored)
        cache_l1_size_mb: int = 256,
        cache_l2_size_mb: int = 512,
        enable_l3_cache: bool = True,
        sync_batch_size: int = 10,
    ):
        """
        Initialize trainable embeddings with simple disk persistence

        Args:
            sample_ids: List of unique sample identifiers
            embedding_dim: Dimension of embedding vectors
            device: Device to store embeddings on (only used for memory mode)
            initialization_strategy: How to initialize embeddings ('zeros', 'normal', 'uniform')
            storage_mode: "memory" (all in RAM) or "disk" (load per chunk)
            embeddings_dir: Directory for embedding chunk files
            chunk_size: Size of each chunk (if None, uses feature manager's chunk size)
            feature_manager: FeatureManager to get chunk structure from
            auto_sync: Whether to automatically sync to disk (always True now)
        """
        self.sample_ids = sorted(sample_ids)
        self.embedding_dim = embedding_dim
        self.device = device
        self.storage_mode = storage_mode

        # Setup disk storage directory
        if embeddings_dir is None:
            import tempfile

            self.embeddings_dir = Path(tempfile.mkdtemp(prefix="embeddings_"))
        else:
            self.embeddings_dir = Path(embeddings_dir)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Determine chunk structure
        if chunk_size is None and feature_manager is not None:
            chunk_size = feature_manager.chunk_size
        self.chunk_size = chunk_size if chunk_size is not None else 1000

        # Build chunk mapping: {chunk_id: [sample_ids]}
        self.chunk_mapping = self._build_chunk_mapping()

        # Build fast lookup: {sample_id: (chunk_id, index_in_chunk)}
        self.id_to_chunk_index = self._build_id_to_chunk_index()

        # Initialize embeddings
        if storage_mode == "memory":
            # Memory mode: keep all embeddings in RAM
            init_tensor = self._initialize_embeddings(initialization_strategy)
            self.embeddings = nn.Parameter(init_tensor.to(device), requires_grad=True)
            self.id_to_index = {sid: idx for idx, sid in enumerate(self.sample_ids)}
            # Save initial state to disk
            self._save_all_to_disk()
        elif storage_mode == "disk":
            # Disk mode: load on demand
            self.embeddings = None
            self.id_to_index = None

            # Check if existing chunks should be loaded
            existing_chunks = list(self.embeddings_dir.glob("embeddings_chunk_*.pt"))
            if existing_chunks:
                print(f"Loading from existing {len(existing_chunks)} chunk files...")
            else:
                # Initialize new chunks on disk
                self._initialize_disk_chunks(initialization_strategy)
        else:
            raise ValueError(f"Unknown storage_mode: {storage_mode}")

        print(f"TrainableEmbeddingManager initialized (NO CACHE):")
        print(f"  - Storage mode: {storage_mode}")
        print(f"  - Total samples: {len(self.sample_ids)}")
        print(f"  - Chunk size: {self.chunk_size}")
        print(f"  - Total chunks: {len(self.chunk_mapping)}")

    def _build_chunk_mapping(self) -> dict:
        """Build mapping from chunk_id to list of sample_ids"""
        chunk_mapping = {}
        for sample_id in self.sample_ids:
            chunk_id = sample_id // self.chunk_size
            if chunk_id not in chunk_mapping:
                chunk_mapping[chunk_id] = []
            chunk_mapping[chunk_id].append(sample_id)
        return chunk_mapping

    def _build_id_to_chunk_index(self) -> dict:
        """Build fast lookup from sample_id to (chunk_id, index_in_chunk)"""
        id_to_chunk_index = {}
        for chunk_id, sample_ids in self.chunk_mapping.items():
            for idx, sample_id in enumerate(sample_ids):
                id_to_chunk_index[sample_id] = (chunk_id, idx)
        return id_to_chunk_index

    def _initialize_embeddings(self, strategy: str) -> torch.Tensor:
        """Initialize embeddings tensor based on strategy"""
        n_samples = len(self.sample_ids)

        if strategy == "zeros":
            return torch.zeros(n_samples, self.embedding_dim)
        elif strategy == "normal":
            return torch.randn(n_samples, self.embedding_dim) * 0.02
        elif strategy == "uniform":
            return torch.rand(n_samples, self.embedding_dim) * 0.02 - 0.01
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def _initialize_disk_chunks(self, strategy: str):
        """Initialize embedding chunks on disk"""
        print(
            f"Initializing {len(self.chunk_mapping)} chunks with strategy: {strategy}"
        )

        for chunk_id, sample_ids in self.chunk_mapping.items():
            chunk_size = len(sample_ids)

            # Initialize embeddings for this chunk
            if strategy == "zeros":
                chunk_embeddings = torch.zeros(chunk_size, self.embedding_dim)
            elif strategy == "normal":
                chunk_embeddings = torch.randn(chunk_size, self.embedding_dim) * 0.02
            elif strategy == "uniform":
                chunk_embeddings = (
                    torch.rand(chunk_size, self.embedding_dim) * 0.02 - 0.01
                )
            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")

            # Create dict mapping sample_id -> embedding
            chunk_dict = {sample_ids[i]: chunk_embeddings[i] for i in range(chunk_size)}

            # Save to disk
            chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
            torch.save(chunk_dict, chunk_path)

    def _save_all_to_disk(self):
        """Save all embeddings to disk in chunks (memory mode)"""
        if self.storage_mode != "memory":
            return

        for chunk_id, sample_ids in self.chunk_mapping.items():
            indices = [self.id_to_index[sid] for sid in sample_ids]
            chunk_embeddings = self.embeddings.data[indices].cpu()

            chunk_dict = {
                sample_ids[i]: chunk_embeddings[i] for i in range(len(sample_ids))
            }

            chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
            torch.save(chunk_dict, chunk_path)

    # ==================== Core Access Methods ====================

    def get_embeddings_by_chunk(self, chunk_id: int) -> Tuple[List[int], torch.Tensor]:
        """
        Get all embeddings from a specific chunk

        Returns:
            (sample_ids, embeddings_tensor)
        """
        if chunk_id not in self.chunk_mapping:
            raise ValueError(f"Chunk {chunk_id} not found")

        chunk_sample_ids = self.chunk_mapping[chunk_id]

        if self.storage_mode == "memory":
            # Get from in-memory tensor
            indices = [self.id_to_index[sid] for sid in chunk_sample_ids]
            chunk_embeddings = self.embeddings[indices]
        else:
            # Load from disk
            chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
            chunk_dict = torch.load(chunk_path, map_location="cpu")

            # Convert dict to tensor in correct order
            chunk_embeddings = torch.stack(
                [chunk_dict[sid] for sid in chunk_sample_ids]
            )

        return chunk_sample_ids, chunk_embeddings

    def update_embeddings_by_chunk(
        self, chunk_id: int, sample_ids: List[int], new_embeddings: torch.Tensor
    ):
        """
        Update all embeddings in a specific chunk

        Args:
            chunk_id: The chunk ID to update
            sample_ids: List of sample IDs in the chunk
            new_embeddings: Tensor of shape (len(sample_ids), embedding_dim)
        """
        if len(sample_ids) != new_embeddings.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(sample_ids)} sample_ids vs {new_embeddings.shape[0]} embeddings"
            )

        if self.storage_mode == "memory":
            # Update in-memory tensor
            indices = [self.id_to_index[sid] for sid in sample_ids]
            with torch.no_grad():
                self.embeddings.data[indices] = new_embeddings.to(self.device)

            # Also update on disk
            chunk_embeddings = new_embeddings.detach().cpu()
            chunk_dict = {
                sample_ids[i]: chunk_embeddings[i] for i in range(len(sample_ids))
            }
            chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
            torch.save(chunk_dict, chunk_path)

        else:
            # Disk mode: load, update, save
            chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"

            # Load existing chunk
            if chunk_path.exists():
                chunk_dict = torch.load(chunk_path, map_location="cpu")
            else:
                chunk_dict = {}

            # Update embeddings
            for i, sample_id in enumerate(sample_ids):
                chunk_dict[sample_id] = new_embeddings[i].detach().cpu()

            # print(chunk_dict)

            # Save back to disk immediately
            torch.save(chunk_dict, chunk_path)

    def get_embeddings(self, sample_ids: List[int]) -> torch.Tensor:
        """
        Get embeddings for specific sample IDs (can be from different chunks)

        Returns:
            Tensor of shape (len(sample_ids), embedding_dim)
        """
        if self.storage_mode == "memory":
            indices = [self.id_to_index[sid] for sid in sample_ids]
            return self.embeddings[indices]
        else:
            # Group by chunks for efficient loading
            chunks_to_load = {}
            for sid in sample_ids:
                chunk_id, _ = self.id_to_chunk_index[sid]
                if chunk_id not in chunks_to_load:
                    chunks_to_load[chunk_id] = []
                chunks_to_load[chunk_id].append(sid)

            # Load and collect embeddings
            sid_to_emb = {}
            for chunk_id, chunk_sample_ids in chunks_to_load.items():
                chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
                chunk_dict = torch.load(chunk_path, map_location="cpu")

                for sid in chunk_sample_ids:
                    sid_to_emb[sid] = chunk_dict[sid]

            # Reorder to match input sample_ids order
            ordered_embeddings = [sid_to_emb[sid] for sid in sample_ids]
            return torch.stack(ordered_embeddings)

    def get_all_embeddings(self) -> Tuple[List[int], torch.Tensor]:
        """
        Get all embeddings

        Returns:
            (all_sample_ids, all_embeddings_tensor)
        """
        if self.storage_mode == "memory":
            return self.sample_ids, self.embeddings.data.cpu()
        else:
            # Load all chunks
            all_embeddings = []
            for chunk_id in sorted(self.chunk_mapping.keys()):
                _, chunk_embeddings = self.get_embeddings_by_chunk(chunk_id)
                all_embeddings.append(chunk_embeddings)

            return self.sample_ids, torch.cat(all_embeddings, dim=0)

    # ==================== Initialization Methods ====================

    def initialize_embeddings_imgtxt(
        self, feature_manager, model, device="cpu", factor=1
    ):
        """
        Initialize label embeddings using imgtxt strategy: (img_emb - txt_emb) @ w_t

        Args:
            feature_manager: FeatureManager instance to load image/text features
            model: CoSiR model to get the projection matrix w_t
            device: Device for computation
        """
        print("##########Initializing label embeddings using imgtxt strategy##########")

        # Get the projection matrix from model
        w_t = model.combiner.label_proj_layer.weight.data.clone().detach().to(device)

        # Get total number of chunks from feature manager
        num_chunks = feature_manager.get_num_chunks()

        # Initialize embeddings chunk by chunk
        for chunk_id in range(num_chunks):
            # Load features for this chunk
            features_data = feature_manager.get_features_by_chunk(chunk_id)

            img_features = features_data["img_features"].to(device)
            txt_features = features_data["txt_features"].to(device)

            # DEBUG: This needs to get from the embedding manager, not the feature manager, still dont know what is the difference between them
            chunk_sample_ids, _ = self.get_embeddings_by_chunk(chunk_id)

            # Compute: (img_emb - txt_emb) @ w_t
            clip_diff = img_features - txt_features
            label_embedding_init = clip_diff @ w_t * factor  # TODO: Here is the factor

            # Update embeddings for this chunk (auto-saves to disk)
            self.update_embeddings_by_chunk(
                chunk_id, chunk_sample_ids, label_embedding_init
            )

            # if chunk_id % 10 == 0 and chunk_id >= 10:
            #     chunk_sample_ids_updated, label_embedding_updated = (
            #         self.get_embeddings_by_chunk(chunk_id)
            #     )
            #     print(label_embedding_init[0])
            #     print(label_embedding_updated[0])

            # Log progress
            if chunk_id % 100 == 0 or chunk_id == num_chunks - 1:
                label_move_distance = torch.norm(
                    label_embedding_init, p=2, dim=1
                ).mean()
                print(
                    f"Chunk: {chunk_id} / {num_chunks-1}, Label Move Distance: {label_move_distance:.3f}"
                )

        print(
            f"✅ Completed imgtxt initialization for {len(self.sample_ids)} samples across {num_chunks} chunks"
        )

    def load_imgtxt_template(self):
        """Load embeddings from template directory"""
        template_dir = self.embeddings_dir.parent.parent / "template_embeddings"

        if not template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        template_chunks = list(template_dir.glob("embeddings_chunk_*.pt"))
        if not template_chunks:
            raise FileNotFoundError(f"No chunk files found in: {template_dir}")

        print(f"Loading template embeddings from: {template_dir}")
        print(f"Found {len(template_chunks)} template chunks")

        # Copy template chunks to current directory
        for template_chunk in template_chunks:
            dest_path = self.embeddings_dir / template_chunk.name
            shutil.copy2(template_chunk, dest_path)

        # If memory mode, reload into memory
        if self.storage_mode == "memory":
            all_embeddings = []
            for chunk_id in sorted(self.chunk_mapping.keys()):
                chunk_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
                chunk_dict = torch.load(chunk_path, map_location="cpu")

                chunk_sample_ids = self.chunk_mapping[chunk_id]
                chunk_embeddings = torch.stack(
                    [chunk_dict[sid] for sid in chunk_sample_ids]
                )
                all_embeddings.append(chunk_embeddings)

            all_embeddings_tensor = torch.cat(all_embeddings, dim=0).to(self.device)
            self.embeddings = nn.Parameter(all_embeddings_tensor, requires_grad=True)

        print(f"✅ Loaded template embeddings for {len(self.sample_ids)} samples")

    def store_imgtxt_template(self):
        """Store current embeddings as template"""
        template_dir = self.embeddings_dir.parent.parent / "template_embeddings"
        template_dir.mkdir(parents=True, exist_ok=True)

        print(f"Storing template embeddings to: {template_dir}")

        # Copy all chunk files to template directory
        for chunk_id in self.chunk_mapping.keys():
            src_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
            dest_path = template_dir / f"embeddings_chunk_{chunk_id}.pt"

            if src_path.exists():
                shutil.copy2(src_path, dest_path)

        print(f"✅ Stored template with {len(self.chunk_mapping)} chunks")

    def save_final_embeddings(self, save_dir: str):
        """Save final embeddings to a specified directory"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Copy all chunk files
        for chunk_id in self.chunk_mapping.keys():
            src_path = self.embeddings_dir / f"embeddings_chunk_{chunk_id}.pt"
            dest_path = save_path / f"embeddings_chunk_{chunk_id}.pt"

            if src_path.exists():
                shutil.copy2(src_path, dest_path)

        print(f"✅ Saved final embeddings to: {save_path}")

    # ==================== Compatibility Methods ====================

    def optimize_cache_settings(self, batch_size: int):
        """Compatibility method - does nothing (no cache)"""
        pass
