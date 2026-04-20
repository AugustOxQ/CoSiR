"""
TrainableEmbeddingManager: per-sample trainable embeddings with numpy-memmap storage.

Storage layout:
    embeddings_dir/
        embeddings.npy   # numpy memmap float32 [N, D]
        sample_ids.npy   # numpy memmap int64   [N]
        metadata.json    # N, embedding_dim

Random access by sample_id is O(1) via a position dict — no chunk files, no dict
loads, no duplication between memory and disk paths.

Modes:
    'ram'  – full tensor loaded into RAM at init; zero disk I/O during training.
    'mmap' – file stays on disk; OS page cache warms it up automatically.
             Use when embeddings are too large for RAM.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm


class TrainableEmbeddingManager:
    """
    Manages per-sample trainable label embeddings.

    After construction the embeddings live in self._data [N, D] (a torch.Tensor
    in 'ram' mode or a numpy memmap in 'mmap' mode).  All access goes through
    get_embeddings / update_embeddings which index by sample_id.
    """

    def __init__(
        self,
        sample_ids: List[int],
        embedding_dim: int,
        embeddings_dir: str,
        mode: str = "ram",
        initialization_strategy: str = "zeros",
        device: str = "cpu",
        # Legacy / compat params (accepted but ignored)
        storage_mode: Optional[str] = None,
        chunk_size: Optional[int] = None,
        feature_manager: Optional[object] = None,
        auto_sync: bool = True,
        cache_l1_size_mb: int = 256,
        cache_l2_size_mb: int = 512,
        enable_l3_cache: bool = True,
        sync_batch_size: int = 10,
    ):
        """
        Args:
            sample_ids:              list of unique integer sample IDs (any order).
            embedding_dim:           dimension of each embedding vector.
            embeddings_dir:          directory where the memmap files are stored.
            mode:                    'ram' or 'mmap'  (default: 'ram').
            initialization_strategy: 'zeros' | 'normal' | 'uniform'.
                                     PCA-based strategies ('imgtxt', 'txt', 'img')
                                     are set via initialize() after construction.
            device:                  torch device for in-memory tensors (ram mode).
        """
        # Compat: honour old 'storage_mode' kwarg if passed
        if storage_mode is not None and mode == "ram":
            mode = "ram" if storage_mode == "memory" else "mmap"

        self.embedding_dim = embedding_dim
        self.device = device
        self.mode = mode
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._emb_path = self.embeddings_dir / "embeddings.npy"
        self._ids_path = self.embeddings_dir / "sample_ids.npy"
        self._meta_path = self.embeddings_dir / "metadata.json"

        # Ordered sample list and O(1) position lookup
        self.sample_ids: List[int] = list(sample_ids)
        self._id_to_pos: Dict[int, int] = {
            sid: pos for pos, sid in enumerate(self.sample_ids)
        }
        N = len(self.sample_ids)

        if self._emb_path.exists() and self._ids_path.exists():
            print(f"[EmbeddingManager] Loading existing embeddings ({N:,} samples)…")
            self._load_storage()
        else:
            print(
                f"[EmbeddingManager] Initialising embeddings "
                f"({N:,} × {embedding_dim}d, strategy={initialization_strategy})…"
            )
            self._create_storage(initialization_strategy)

        print(
            f"[EmbeddingManager] Ready  mode={self.mode}  "
            f"N={N:,}  D={embedding_dim}"
        )

    # ── Storage helpers ────────────────────────────────────────────────────────

    def _create_storage(self, strategy: str) -> None:
        """Allocate and initialise memmap files from scratch."""
        N = len(self.sample_ids)
        D = self.embedding_dim

        # Write sample_ids
        ids_mm = np.lib.format.open_memmap(
            self._ids_path, mode="w+", dtype=np.int64, shape=(N,)
        )
        ids_mm[:] = np.array(self.sample_ids, dtype=np.int64)
        ids_mm.flush()
        del ids_mm

        # Write embeddings
        emb_mm = np.lib.format.open_memmap(
            self._emb_path, mode="w+", dtype=np.float32, shape=(N, D)
        )
        emb_mm[:] = self._make_init_array(N, D, strategy)
        emb_mm.flush()
        del emb_mm

        self._save_metadata(N, D)
        self._load_storage()

    def _load_storage(self) -> None:
        """Open existing memmap files and (optionally) pull into RAM."""
        mm = np.lib.format.open_memmap(self._emb_path, mode="r+")
        if self.mode == "ram":
            new_data = torch.from_numpy(np.array(mm)).to(self.device)
            if hasattr(self, "_data") and isinstance(self._data, nn.Parameter):
                # Update in-place so any existing optimizer reference stays valid
                self._data.data.copy_(new_data)
            else:
                self._data = nn.Parameter(new_data)
                self.embeddings: nn.Parameter = self._data
                self.id_to_index: Dict[int, int] = self._id_to_pos
        else:
            self._mmap = mm  # keep alive; access via torch.from_numpy slice

    def _save_metadata(self, N: int, D: int) -> None:
        with open(self._meta_path, "w") as f:
            json.dump({"total_samples": N, "embedding_dim": D}, f, indent=2)

    @staticmethod
    def _make_init_array(N: int, D: int, strategy: str) -> np.ndarray:
        if strategy == "zeros":
            return np.zeros((N, D), dtype=np.float32)
        elif strategy == "normal":
            return (np.random.randn(N, D) * 0.02).astype(np.float32)
        elif strategy == "uniform":
            return (np.random.rand(N, D) * 0.02 - 0.01).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown initialization_strategy '{strategy}'. "
                "Use 'zeros', 'normal', or 'uniform'. "
                "For PCA-based init call initialize() separately."
            )

    # ── Core access ────────────────────────────────────────────────────────────

    def get_embeddings(self, sample_ids: List[int]) -> torch.Tensor:
        """
        Return embeddings for the given sample IDs as a float32 tensor [B, D].
        Order matches the input list.
        """
        positions = [self._id_to_pos[sid] for sid in sample_ids]
        if self.mode == "ram":
            return self._data[positions]
        else:
            arr = self._mmap[positions]
            return torch.from_numpy(np.array(arr))

    def update_embeddings(
        self, sample_ids: List[int], new_embeddings: torch.Tensor
    ) -> None:
        """
        Write updated embeddings back to storage.

        Args:
            sample_ids:     IDs whose embeddings changed (same order as new_embeddings).
            new_embeddings: tensor [B, D].
        """
        positions = [self._id_to_pos[sid] for sid in sample_ids]
        data_np = new_embeddings.detach().cpu().float().numpy()

        if self.mode == "ram":
            self._data.data[positions] = torch.from_numpy(data_np).to(self.device)

        # Always persist to disk so checkpoints are consistent
        mm = np.lib.format.open_memmap(self._emb_path, mode="r+")
        mm[positions] = data_np
        mm.flush()
        del mm

    def get_all_embeddings(self) -> Tuple[List[int], torch.Tensor]:
        """
        Return (sample_ids_list, all_embeddings [N, D]).
        Used by evaluation / clustering code.
        """
        if self.mode == "ram":
            return self.sample_ids, self._data.detach().cpu()
        else:
            arr = np.lib.format.open_memmap(self._emb_path, mode="r")
            return self.sample_ids, torch.from_numpy(np.array(arr))

    def _save_all_chunks_to_disk(self) -> None:
        """Flush in-memory embeddings (RAM mode) to the memmap file on disk."""
        if self.mode != "ram":
            return
        mm = np.lib.format.open_memmap(self._emb_path, mode="r+")
        mm[:] = self._data.data.cpu().float().numpy()
        mm.flush()
        del mm

    # ── PCA-based initialisation ───────────────────────────────────────────────

    def initialize(
        self,
        strategy: str,
        feature_manager=None,
        model=None,
        device: str = "cpu",
        factor: float = 1.0,
        normalize: bool = False,
    ) -> None:
        """
        (Re-)initialise embeddings.  Handles all strategies in one place.

        Simple strategies ('zeros', 'normal', 'uniform') do not need
        feature_manager.  PCA-based strategies ('imgtxt', 'txt', 'img')
        require it.

        Args:
            strategy:        one of 'zeros'|'normal'|'uniform'|'imgtxt'|'txt'|'img'.
            feature_manager: FeatureManager instance (required for PCA strategies).
            model:           unused — kept for API compatibility.
            device:          compute device for feature loading.
            factor:          scalar multiplier applied after PCA normalisation.
        """
        N = len(self.sample_ids)
        D = self.embedding_dim

        if strategy in ("zeros", "normal", "uniform"):
            data = self._make_init_array(N, D, strategy)
        elif strategy in ("imgtxt", "txt", "img"):
            data = self._pca_init(strategy, feature_manager, device, factor, normalize)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

        # Write to memmap
        mm = np.lib.format.open_memmap(self._emb_path, mode="r+")
        mm[:] = data
        mm.flush()
        del mm

        # Reload into RAM if needed; update in-place to keep optimizer reference valid
        if self.mode == "ram":
            new_data = torch.from_numpy(data).to(self.device)
            if hasattr(self, "_data") and isinstance(self._data, nn.Parameter):
                self._data.data.copy_(new_data)
            else:
                self._data = nn.Parameter(new_data)
                self.embeddings = self._data

        print(f"[EmbeddingManager] Initialised with strategy='{strategy}'")

    def _pca_init(
        self,
        strategy: str,
        feature_manager,
        device: str,
        factor: float,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Load all features from FeatureManager, compute PCA, return normalised array.
        """
        print(f"[EmbeddingManager] PCA init (strategy={strategy})…")

        # Load features shard-by-shard to handle large datasets
        source_parts: List[np.ndarray] = []
        num_shards = feature_manager.get_num_chunks()

        for shard_id in tqdm(range(num_shards), desc="Loading features"):
            feats = feature_manager.get_features_by_chunk(shard_id)
            img = feats["img_features"].to(device)
            txt = feats["txt_features"].to(device)

            if strategy == "imgtxt":
                part = (img - txt).cpu().numpy()
            elif strategy == "txt":
                part = txt.cpu().numpy()
            else:  # 'img'
                part = img.cpu().numpy()

            if normalize:
                part = part / np.linalg.norm(part, axis=-1, keepdims=True)

            source_parts.append(part)

        source = np.concatenate(source_parts, axis=0)  # [N, clip_dim]
        print(f"[EmbeddingManager] Running PCA: {source.shape} → {self.embedding_dim}d")

        pca = PCA(n_components=self.embedding_dim)
        projected = pca.fit_transform(source).astype(np.float32)  # [N, D]

        # Normalise to zero mean, unit std, then scale
        projected = (projected - projected.mean(0)) / (projected.std(0) + 1e-8)
        projected = (projected * factor).astype(np.float32)

        # The feature manager iterates shards in order 0..K-1.
        # We need to re-order rows to match self.sample_ids order.
        shard_sample_ids: List[int] = feature_manager.get_all_sample_ids()
        fm_pos = {sid: i for i, sid in enumerate(shard_sample_ids)}
        reorder = [fm_pos[sid] for sid in self.sample_ids]
        projected = projected[reorder]

        print(
            f"[EmbeddingManager] PCA init done. "
            f"Mean norm: {np.linalg.norm(projected, axis=1).mean():.4f}"
        )

        if normalize:
            projected = projected / np.linalg.norm(projected, axis=-1, keepdims=True)

        return projected

    # ── Template persistence ───────────────────────────────────────────────────

    def _copy_to(self, dest_dir: Path) -> None:
        """Copy memmap files to dest_dir."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("embeddings.npy", "sample_ids.npy", "metadata.json"):
            src = self.embeddings_dir / fname
            if src.exists():
                shutil.copy2(src, dest_dir / fname)

    def _copy_from(self, src_dir: Path) -> None:
        """Replace current memmap files with copies from src_dir, then reload."""
        for fname in ("embeddings.npy", "sample_ids.npy", "metadata.json"):
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, self.embeddings_dir / fname)
        self._load_storage()

    def store_imgtxt_template(self) -> None:
        """Save current embeddings as a reusable template."""
        template_dir = self.embeddings_dir.parent.parent / "template_embeddings"
        self._copy_to(template_dir)
        print(f"[EmbeddingManager] Template saved → {template_dir}")

    def load_imgtxt_template(self) -> None:
        """Load embeddings from the sibling template_embeddings directory."""
        template_dir = self.embeddings_dir.parent.parent / "template_embeddings"
        if not template_dir.exists() or not (template_dir / "embeddings.npy").exists():
            raise FileNotFoundError(f"No template found at {template_dir}")
        self._copy_from(template_dir)
        print(f"[EmbeddingManager] Template loaded ← {template_dir}")

    def load_phase_1_template(self, path: str) -> None:
        """Load embeddings from an arbitrary directory (e.g. phase-1 output)."""
        src_dir = Path(path)
        if not src_dir.exists() or not (src_dir / "embeddings.npy").exists():
            raise FileNotFoundError(f"No embeddings found at {src_dir}")
        self._copy_from(src_dir)
        print(f"[EmbeddingManager] Embeddings loaded ← {src_dir}")

    # ── Convenience wrappers kept for training-script compatibility ────────────

    def initialize_embeddings_imgtxt(
        self, feature_manager, model=None, device="cpu", factor=1, normalize=False
    ) -> None:
        self.initialize("imgtxt", feature_manager, model, device, factor, normalize)

    def initialize_embeddings_txt(
        self, feature_manager, model=None, device="cpu", factor=1, normalize=False
    ) -> None:
        self.initialize("txt", feature_manager, model, device, factor, normalize)

    def initialize_embeddings_img(
        self, feature_manager, model=None, device="cpu", factor=1, normalize=False
    ) -> None:
        self.initialize("img", feature_manager, model, device, factor, normalize)

    def optimize_cache_settings(self, batch_size: int) -> None:
        """No-op — kept for call-site compatibility."""
        pass
