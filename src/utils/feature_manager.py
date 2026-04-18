"""
FeatureManager: Shard-based storage for pre-extracted CLIP features.

Storage layout:
    storage_dir/
        metadata.json           # total_samples, num_shards, shard_size,
                                # feature_dims, available_features
        shards/
            shard_00000/
                img_features.npy    # np.lib.format memmap  [n, D]   float32
                txt_features.npy    # np.lib.format memmap  [n, D]   float32
                sample_ids.npy      # np.lib.format memmap  [n]      int64
                img_full.h5         # optional HDF5 + gzip  [n, ...]
                txt_full.h5         # optional HDF5 + gzip  [n, ...]
            shard_00001/
                ...

Training strategy (auto-selected at runtime):
    RAM mode:    CLS features fit in available RAM → load all to RAM, use
                 DataLoader(shuffle=True) → true per-epoch random batches.
    Stream mode: Dataset too large → yield shards in shuffled order with an
                 in-memory shuffle window → "sufficient" randomness.

Compatibility note:
    get_num_chunks() and get_features_by_chunk() are kept as thin shims so
    existing EmbeddingManager code continues to work without changes.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import psutil
import torch


# Which feature names live in per-shard numpy memmaps
_MEMMAP_NAMES = frozenset({"img_features", "txt_features", "sample_ids"})
# Which feature names live in per-shard HDF5 files
_HDF5_NAMES = frozenset({"img_full", "txt_full"})


class FeatureManager:
    """
    Manages pre-extracted CLIP features using shard-based binary storage.
    Write once during extraction; read many times during training.
    """

    DEFAULT_SHARD_SIZE = 100_000

    def __init__(
        self,
        storage_dir: str,
        shard_size: int = DEFAULT_SHARD_SIZE,
        hdf5_compression: bool = True,
        hdf5_compression_level: int = 4,
        # Legacy compat params (ignored)
        chunk_size: int = 2048,
        config: Optional[Dict[str, Any]] = None,
        preload_index: bool = False,
    ):
        self.storage_dir = Path(storage_dir)
        self.shards_dir = self.storage_dir / "shards"
        self.metadata_path = self.storage_dir / "metadata.json"
        self.shard_size = shard_size
        self.hdf5_compression = hdf5_compression
        self.hdf5_compression_level = hdf5_compression_level

        self._metadata: Optional[Dict[str, Any]] = None
        self._write_cursor: int = 0  # global position during extraction

    # ── Metadata ───────────────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"No feature store found at '{self.storage_dir}'. "
                    "Run feature extraction first (set load_existing_features: false)."
                )
            with open(self.metadata_path) as f:
                self._metadata = json.load(f)
        return self._metadata

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def validate_backbone(self, expected_model: str) -> None:
        """
        Verify that the stored features were extracted with ``expected_model``.

        Raises:
            ValueError: if the stored backbone does not match ``expected_model``.

        Old stores that pre-date backbone tracking emit a warning and continue
        (backward compatibility), but you should re-extract when convenient.
        """
        stored = self.metadata.get("backbone_model")
        if stored is None:
            print(
                f"[FeatureManager] WARNING: feature store at '{self.storage_dir}' "
                f"has no backbone_model recorded (old store). "
                f"Assuming it matches '{expected_model}'. "
                "Re-extract to silence this warning."
            )
            return
        if stored != expected_model:
            raise ValueError(
                f"[FeatureManager] Backbone mismatch!\n"
                f"  stored  : {stored}\n"
                f"  current : {expected_model}\n"
                f"The cached features at '{self.storage_dir}' were extracted with a "
                f"different backbone. Delete the feature store and re-run extraction "
                f"with the current model config."
            )

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def total_samples(self) -> int:
        return self.metadata["total_samples"]

    @property
    def num_shards(self) -> int:
        return self.metadata["num_shards"]

    @property
    def available_features(self) -> List[str]:
        """Feature names stored in this store, excluding 'sample_ids'."""
        return self.metadata["available_features"]

    def __len__(self) -> int:
        return self.total_samples

    def _shard_dir(self, shard_id: int) -> Path:
        return self.shards_dir / f"shard_{shard_id:05d}"

    def _shard_len(self, shard_id: int) -> int:
        """Number of samples in a specific shard (last one may be smaller)."""
        ss = self.metadata["shard_size"]
        total = self.metadata["total_samples"]
        n_full = total // ss
        if shard_id < n_full:
            return ss
        remainder = total % ss
        return remainder if remainder else ss

    def _shard_len_pre(self, shard_id: int, total_samples: int) -> int:
        """Same as _shard_len but usable before metadata is stored."""
        n_full = total_samples // self.shard_size
        if shard_id < n_full:
            return self.shard_size
        remainder = total_samples % self.shard_size
        return remainder if remainder else self.shard_size

    # ── Write phase ────────────────────────────────────────────────────────────

    def open_for_writing(
        self,
        total_samples: int,
        feature_dims: Dict[str, Tuple[int, ...]],
        backbone_model: Optional[str] = None,
    ) -> None:
        """
        Initialise the shard files before extraction begins.

        Args:
            total_samples:  Exact number of samples that will be written.
            feature_dims:   Maps feature name → shape tuple (batch dim excluded).
                            Must include "img_features" and "txt_features".
                            "img_full" / "txt_full" are optional — omit them if
                            the extraction dataset does not produce them.
            backbone_model: HuggingFace model ID used to extract the features
                            (e.g. "openai/clip-vit-base-patch32").  Stored in
                            metadata and validated by validate_backbone() on load.
        Raises:
            FileExistsError: if a store already exists in storage_dir.
        """
        if self.metadata_path.exists():
            raise FileExistsError(
                f"A feature store already exists at '{self.storage_dir}'. "
                "Delete the directory (or its metadata.json) before re-extracting."
            )

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        num_shards = math.ceil(total_samples / self.shard_size)

        self._metadata = {
            "total_samples": total_samples,
            "num_shards": num_shards,
            "shard_size": self.shard_size,
            "feature_dims": {k: list(v) for k, v in feature_dims.items()},
            "available_features": list(feature_dims.keys()),
            "hdf5_compression": self.hdf5_compression,
            "hdf5_compression_level": self.hdf5_compression_level,
            "backbone_model": backbone_model,
        }
        self._write_cursor = 0

        for shard_id in range(num_shards):
            n = self._shard_len_pre(shard_id, total_samples)
            shard_dir = self._shard_dir(shard_id)
            shard_dir.mkdir(parents=True, exist_ok=True)
            self._preallocate_shard(shard_id, n, feature_dims)

        self._save_metadata()
        print(
            f"Opened feature store for writing: {total_samples:,} samples, "
            f"{num_shards} shards of up to {self.shard_size:,} samples each."
        )

    def _preallocate_shard(
        self,
        shard_id: int,
        n: int,
        feature_dims: Dict[str, Tuple[int, ...]],
    ) -> None:
        shard_dir = self._shard_dir(shard_id)

        # CLS features + sample_ids as numpy memmaps (.npy format)
        for feat_name, dtype, default_dim in [
            ("img_features", np.float32, None),
            ("txt_features", np.float32, None),
        ]:
            if feat_name not in feature_dims:
                continue
            dims = tuple(feature_dims[feat_name])
            np.lib.format.open_memmap(
                shard_dir / f"{feat_name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(n,) + dims,
            )

        np.lib.format.open_memmap(
            shard_dir / "sample_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n,),
        )

        # Full-sequence features as HDF5 + gzip
        compress_kw: Dict[str, Any] = (
            {"compression": "gzip", "compression_opts": self.hdf5_compression_level}
            if self.hdf5_compression
            else {}
        )
        for feat_name in ("img_full", "txt_full"):
            if feat_name not in feature_dims:
                continue
            dims = tuple(feature_dims[feat_name])
            chunk_shape = (min(256, n),) + dims
            with h5py.File(shard_dir / f"{feat_name}.h5", "w") as f:
                f.create_dataset(
                    "data",
                    shape=(n,) + dims,
                    dtype="float32",
                    chunks=chunk_shape,
                    **compress_kw,
                )

    def write_batch(
        self,
        img_features: torch.Tensor,
        txt_features: torch.Tensor,
        sample_ids: List[int],
        img_full: Optional[torch.Tensor] = None,
        txt_full: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Append one batch of features.  Handles shard boundaries automatically.
        Call repeatedly during extraction; call finalize_writing() when done.
        """
        img_np = img_features.detach().cpu().float().numpy()
        txt_np = txt_features.detach().cpu().float().numpy()
        ids_np = np.array(sample_ids, dtype=np.int64)
        img_full_np = (
            img_full.detach().cpu().float().numpy() if img_full is not None else None
        )
        txt_full_np = (
            txt_full.detach().cpu().float().numpy() if txt_full is not None else None
        )

        n_batch = len(ids_np)
        written = 0

        while written < n_batch:
            shard_id = self._write_cursor // self.shard_size
            pos = self._write_cursor % self.shard_size
            shard_cap = self._shard_len_pre(
                shard_id, self._metadata["total_samples"]
            )
            n_here = min(n_batch - written, shard_cap - pos)
            end = pos + n_here
            shard_dir = self._shard_dir(shard_id)

            sl = slice(written, written + n_here)

            # Memmap writes
            for fname, data in (
                ("img_features.npy", img_np[sl]),
                ("txt_features.npy", txt_np[sl]),
                ("sample_ids.npy", ids_np[sl]),
            ):
                path = shard_dir / fname
                if path.exists():
                    mm = np.lib.format.open_memmap(path, mode="r+")
                    mm[pos:end] = data
                    mm.flush()
                    del mm

            # HDF5 writes
            if img_full_np is not None:
                path = shard_dir / "img_full.h5"
                if path.exists():
                    with h5py.File(path, "r+") as f:
                        f["data"][pos:end] = img_full_np[sl]

            if txt_full_np is not None:
                path = shard_dir / "txt_full.h5"
                if path.exists():
                    with h5py.File(path, "r+") as f:
                        f["data"][pos:end] = txt_full_np[sl]

            written += n_here
            self._write_cursor += n_here

    def finalize_writing(self) -> None:
        """Call after the last write_batch to update metadata with actual count."""
        actual = self._write_cursor
        expected = self._metadata["total_samples"]
        if actual != expected:
            print(
                f"Warning: expected {expected} samples but wrote {actual}. "
                "Updating metadata."
            )
            self._metadata["total_samples"] = actual
            self._metadata["num_shards"] = math.ceil(actual / self.shard_size)
            self._save_metadata()
        print(
            f"Feature extraction complete: {actual:,} samples in "
            f"{self._metadata['num_shards']} shards → {self.storage_dir}"
        )

    # ── Read phase ─────────────────────────────────────────────────────────────

    def load_shard_to_ram(
        self,
        shard_id: int,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load one complete shard into RAM as torch tensors.

        Args:
            shard_id:      which shard (0 … num_shards-1).
            feature_types: subset of available_features to load;
                           None means all available. "sample_ids" is always
                           included regardless of this argument.
        Returns:
            dict with requested feature tensors plus "sample_ids".
        """
        if feature_types is None:
            feature_types = self.available_features
        to_load = list(dict.fromkeys(list(feature_types) + ["sample_ids"]))

        shard_dir = self._shard_dir(shard_id)
        result: Dict[str, torch.Tensor] = {}

        for name in to_load:
            if name in _MEMMAP_NAMES:
                path = shard_dir / f"{name}.npy"
                if not path.exists():
                    continue
                arr = np.lib.format.open_memmap(path, mode="r")
                result[name] = torch.from_numpy(np.array(arr))  # copy into RAM
            elif name in _HDF5_NAMES:
                path = shard_dir / f"{name}.h5"
                if not path.exists():
                    continue
                with h5py.File(path, "r") as f:
                    result[name] = torch.from_numpy(np.array(f["data"]))

        return result

    def load_all_to_ram(
        self,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate every shard into a single tensor dict.
        Only call this when fits_in_ram() returns True.
        """
        accum: Dict[str, List[torch.Tensor]] = {}
        for shard_id in range(self.num_shards):
            shard = self.load_shard_to_ram(shard_id, feature_types)
            for k, v in shard.items():
                accum.setdefault(k, []).append(v)
        return {k: torch.cat(vs, dim=0) for k, vs in accum.items()}

    def get_all_sample_ids(self) -> List[int]:
        """
        Return a flat list of all sample IDs in storage order
        (shard 0 first, then shard 1, …).
        Used to initialise the EmbeddingManager.
        """
        ids: List[int] = []
        for shard_id in range(self.num_shards):
            path = self._shard_dir(shard_id) / "sample_ids.npy"
            arr = np.lib.format.open_memmap(path, mode="r")
            ids.extend(arr.tolist())
        return ids

    # ── Strategy helpers ───────────────────────────────────────────────────────

    def cls_features_size_gb(self) -> float:
        """Estimated GiB needed to hold img + txt CLS features in RAM."""
        dims = self.metadata.get("feature_dims", {})
        img_d = int(np.prod(dims.get("img_features", [512])))
        txt_d = int(np.prod(dims.get("txt_features", [512])))
        return self.total_samples * (img_d + txt_d) * 4 / (1024 ** 3)

    def fits_in_ram(self, safety_factor: float = 0.6) -> bool:
        """
        True when CLS features fit in available RAM with a safety margin
        (leaving room for model weights, CUDA, activations, etc.).
        """
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return self.cls_features_size_gb() < available_gb * safety_factor

    # ── EmbeddingManager compatibility shims ──────────────────────────────────
    # These allow the existing EmbeddingManager (which iterates over "chunks")
    # to work unchanged.  Shards are treated as chunks.

    def get_num_chunks(self) -> int:
        """Compat: return number of shards (treated as chunks)."""
        return self.num_shards

    def get_features_by_chunk(
        self,
        chunk_id: int,
        feature_types: Optional[List[str]] = None,
        async_prefetch: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compat shim: load shard `chunk_id` and return features dict.
        Signature matches the old FeatureManager API so EmbeddingManager
        initialisation methods work without changes.
        """
        return self.load_shard_to_ram(chunk_id, feature_types)

    # Legacy attribute so old code that reads feature_manager.chunk_size
    # doesn't crash immediately.
    @property
    def chunk_size(self) -> int:
        return self.metadata.get("shard_size", self.shard_size)
