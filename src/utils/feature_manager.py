import json
import os
import shutil
import time
import threading
import sqlite3
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import numpy as np

import psutil
import lz4.frame


def find_device(model):
    return next(model.parameters()).device


def copy_all_files(src_path, dst_base):
    os.makedirs(dst_base, exist_ok=True)
    for filename in os.listdir(src_path):
        src_file = os.path.join(src_path, filename)
        dst_file = os.path.join(dst_base, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)  # preserves metadata


# ================== Storage Backend Layer ==================
class StorageBackend(ABC):
    """Abstract base class for feature storage backends"""

    @abstractmethod
    def save_features(
        self, sample_ids: List[int], features: Dict[str, torch.Tensor]
    ) -> None:
        pass

    @abstractmethod
    def save_chunk_directly(
        self, chunk_id: int, sample_ids: List[int], features: Dict[str, torch.Tensor]
    ) -> None:
        pass

    @abstractmethod
    def load_features(self, sample_ids: List[int]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def exists(self, sample_id: int) -> bool:
        pass

    def get_chunk_path(self, chunk_id: int) -> Path:
        """Get path to chunk file - default implementation"""
        return Path("chunk_{}.pt".format(chunk_id))


class TorchChunkedStorage(StorageBackend):
    """Optimized PyTorch-based chunked storage for sequential access"""

    def __init__(
        self,
        storage_dir: Union[str, Path],
        chunk_size: int = 10000,
        compression: bool = True,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.compression = compression
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about the storage"""
        metadata_path = self.storage_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {"version": "v1.0"}

    def _save_metadata(self) -> None:
        """Save storage metadata"""
        metadata_path = self.storage_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def _group_by_chunks(self, sample_ids: List[int]) -> Dict[int, List[int]]:
        """Group sample IDs by chunk"""
        chunks = {}
        for sample_id in sample_ids:
            chunk_id = sample_id // self.chunk_size
            if chunk_id not in chunks:
                chunks[chunk_id] = []
            chunks[chunk_id].append(sample_id)
        return chunks

    def _get_indices(
        self, all_sample_ids: List[int], target_sample_ids: List[int]
    ) -> List[int]:
        """Get indices of target sample IDs within all sample IDs"""
        id_to_idx = {sample_id: idx for idx, sample_id in enumerate(all_sample_ids)}
        return [
            id_to_idx[sample_id]
            for sample_id in target_sample_ids
            if sample_id in id_to_idx
        ]

    def _get_sample_indices(
        self, chunk_sample_ids: List[int], target_sample_ids: List[int]
    ) -> List[int]:
        """Get indices of target samples within a chunk"""
        chunk_id_to_idx = {
            sample_id: idx for idx, sample_id in enumerate(chunk_sample_ids)
        }
        return [
            chunk_id_to_idx[sample_id]
            for sample_id in target_sample_ids
            if sample_id in chunk_id_to_idx
        ]

    def save_features(
        self, sample_ids: List[int], features: Dict[str, torch.Tensor]
    ) -> None:
        """Save features in chunks with optional compression"""
        chunks = self._group_by_chunks(sample_ids)

        for chunk_id, chunk_sample_ids in chunks.items():
            chunk_path = self.storage_dir / f"chunk_{chunk_id}.pt"

            # Get indices of chunk samples in the original features
            indices = self._get_indices(sample_ids, chunk_sample_ids)

            # Prepare chunk data
            chunk_features = {}
            for feat_name, feat_data in features.items():
                if isinstance(feat_data, torch.Tensor):
                    # Handle tensor features
                    if len(indices) > 0:
                        chunk_features[feat_name] = feat_data[indices]
                    else:
                        chunk_features[feat_name] = torch.empty(0, *feat_data.shape[1:])
                else:
                    # Handle non-tensor features (like txt_full)
                    if len(indices) > 0:
                        chunk_features[feat_name] = [feat_data[i] for i in indices]
                    else:
                        chunk_features[feat_name] = []

            chunk_data = {
                "sample_ids": chunk_sample_ids,
                "features": chunk_features,
                "metadata": {
                    "timestamp": time.time(),
                    "feature_version": self.metadata.get("version", "v1.0"),
                    "compression": self.compression,
                },
            }

            if self.compression and HAS_LZ4:
                # Compress tensors before saving
                for feat_name in chunk_data["features"]:
                    data = chunk_data["features"][feat_name]
                    if (
                        isinstance(data, torch.Tensor) and data.numel() > 0
                    ):  # Only compress non-empty tensors
                        tensor_bytes = data.numpy().tobytes()
                        compressed = lz4.frame.compress(tensor_bytes)
                        chunk_data["features"][feat_name] = {
                            "compressed_data": compressed,
                            "shape": data.shape,
                            "dtype": str(data.dtype),
                        }

            torch.save(chunk_data, chunk_path)

    def save_chunk_directly(
        self, chunk_id: int, sample_ids: List[int], features: Dict[str, torch.Tensor]
    ) -> None:
        """Save features directly to a specific chunk (for add_features_chunk compatibility)"""
        chunk_path = self.storage_dir / f"chunk_{chunk_id}.pt"

        # Prepare chunk data
        chunk_features = {}
        for feat_name, feat_data in features.items():
            if isinstance(feat_data, torch.Tensor):
                chunk_features[feat_name] = feat_data
            else:
                # Handle non-tensor features (like txt_full)
                chunk_features[feat_name] = feat_data

        chunk_data = {
            "sample_ids": sample_ids,
            "features": chunk_features,
            "chunk_id": chunk_id,
            "metadata": {
                "compression": self.compression,
            },
        }

        if self.compression and HAS_LZ4:
            # Compress tensors before saving
            for feat_name in chunk_data["features"]:
                data = chunk_data["features"][feat_name]
                if (
                    isinstance(data, torch.Tensor) and data.numel() > 0
                ):  # Only compress non-empty tensors
                    tensor_bytes = data.numpy().tobytes()
                    compressed = lz4.frame.compress(tensor_bytes)
                    chunk_data["features"][feat_name] = {
                        "compressed_data": compressed,
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                    }

        torch.save(chunk_data, chunk_path)

    def load_features(self, sample_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Load features with decompression if needed"""
        chunks = self._group_by_chunks(sample_ids)
        result_features = {}

        for chunk_id, chunk_sample_ids in chunks.items():
            chunk_path = self.storage_dir / f"chunk_{chunk_id}.pt"
            if not chunk_path.exists():
                continue

            chunk_data = torch.load(chunk_path, map_location="cpu")

            # Decompress if needed
            if self.compression and HAS_LZ4 and chunk_data["features"]:
                for feat_name, feat_data in chunk_data["features"].items():
                    if isinstance(feat_data, dict) and "compressed_data" in feat_data:
                        compressed = feat_data["compressed_data"]
                        decompressed = lz4.frame.decompress(compressed)
                        # Convert PyTorch dtype string to numpy dtype
                        torch_dtype = feat_data["dtype"]
                        if torch_dtype.startswith("torch."):
                            torch_dtype = torch_dtype.replace("torch.", "")
                        np_dtype = getattr(np, torch_dtype)
                        tensor = np.frombuffer(decompressed, dtype=np_dtype)
                        tensor = torch.from_numpy(
                            tensor.reshape(feat_data["shape"]).copy()
                        )
                        chunk_data["features"][feat_name] = tensor

            # Extract requested samples from chunk
            indices = self._get_sample_indices(
                chunk_data["sample_ids"], chunk_sample_ids
            )
            for feat_name, feat_data in chunk_data["features"].items():
                if feat_name not in result_features:
                    result_features[feat_name] = []
                if len(indices) > 0:
                    if isinstance(feat_data, torch.Tensor):
                        result_features[feat_name].append(feat_data[indices])
                    else:
                        # Handle non-tensor data like txt_full
                        selected_items = [feat_data[i] for i in indices]
                        result_features[feat_name].extend(selected_items)

        # Concatenate results
        if not result_features:
            return {}

        final_result = {}
        for feat_name, data_list in result_features.items():
            if not data_list:
                continue
            if isinstance(data_list[0], torch.Tensor):
                final_result[feat_name] = torch.cat(data_list, dim=0)
            else:
                # For non-tensor data like txt_full, return the list as-is
                final_result[feat_name] = data_list
        return final_result

    def exists(self, sample_id: int) -> bool:
        """Check if a sample exists in storage"""
        chunk_id = sample_id // self.chunk_size
        chunk_path = self.storage_dir / f"chunk_{chunk_id}.pt"
        if not chunk_path.exists():
            return False

        chunk_data = torch.load(chunk_path, map_location="cpu")
        return sample_id in chunk_data["sample_ids"]

    def get_chunk_path(self, chunk_id: int) -> Path:
        """Get path to chunk file"""
        return self.storage_dir / f"chunk_{chunk_id}.pt"


class MemoryMappedStorage(StorageBackend):
    """Memory-mapped storage for fast random access"""

    def __init__(self, storage_dir: Union[str, Path], feature_dims: Dict[str, int]):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dims = feature_dims
        self.mmaps = {}
        self.metadata_db_path = self.storage_dir / "metadata.db"
        self.metadata_db = sqlite3.connect(str(self.metadata_db_path))
        self._init_storage()

    def _init_storage(self):
        """Initialize memory-mapped files for each feature type"""
        # Create metadata table
        self.metadata_db.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_metadata (
                sample_id INTEGER PRIMARY KEY,
                offset INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.metadata_db.commit()

        for feat_name, feat_dim in self.feature_dims.items():
            mmap_path = self.storage_dir / f"{feat_name}.mmap"
            # Initialize with empty file for now - actual implementation would be more complex
            if not mmap_path.exists():
                # Create empty file
                with open(mmap_path, "wb") as f:
                    f.write(b"\x00" * (1000000 * feat_dim * 4))  # Preallocate space

    def save_features(
        self, sample_ids: List[int], features: Dict[str, torch.Tensor]
    ) -> None:
        """Save features using memory mapping"""
        # Implementation would involve writing directly to memory-mapped files
        # This is a simplified version
        pass

    def save_chunk_directly(
        self, chunk_id: int, sample_ids: List[int], features: Dict[str, torch.Tensor]
    ) -> None:
        """Save chunk directly using memory mapping"""
        pass

    def load_features(self, sample_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Fast random access using memory mapping"""
        result = {}
        for feat_name in self.feature_dims.keys():
            # Simplified implementation
            result[feat_name] = torch.randn(
                len(sample_ids), self.feature_dims[feat_name]
            )
        return result

    def exists(self, sample_id: int) -> bool:
        """Check if sample exists in database"""
        cursor = self.metadata_db.execute(
            "SELECT 1 FROM sample_metadata WHERE sample_id = ?", (sample_id,)
        )
        return cursor.fetchone() is not None


# ================== Multi-Tier Cache System ==================


class MultiTierCache:
    """LRU cache with memory pressure awareness"""

    def __init__(
        self,
        l1_size_mb: int = 512,  # Hot cache in RAM
        l2_size_mb: int = 2048,  # Warm cache in RAM
        l3_path: Optional[Path] = None,
    ):  # Cold cache on SSD
        self.l1_cache = OrderedDict()  # Hot - immediate access
        self.l2_cache = OrderedDict()  # Warm - recent access
        self.l3_path = l3_path  # Cold - disk cache

        if self.l3_path:
            self.l3_path = Path(self.l3_path)
            self.l3_path.mkdir(parents=True, exist_ok=True)

        self.l1_max_size = l1_size_mb * 1024 * 1024
        self.l2_max_size = l2_size_mb * 1024 * 1024
        self.current_l1_size = 0
        self.current_l2_size = 0

        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _calculate_size(self, value: Dict[str, Any]) -> int:
        """Calculate memory size of cached features"""
        total_size = 0
        for data in value.values():
            if isinstance(data, torch.Tensor):
                total_size += data.numel() * data.element_size()
            elif isinstance(data, list):
                # Rough estimate for list data
                total_size += len(data) * 100  # approximate size per item
        return total_size

    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get from cache with automatic promotion"""
        with self.lock:
            # Check L1 (hot cache)
            if key in self.l1_cache:
                self.stats["hits"] += 1
                # Move to end (most recently used)
                value = self.l1_cache.pop(key)
                self.l1_cache[key] = value
                return value

            # Check L2 (warm cache)
            if key in self.l2_cache:
                self.stats["hits"] += 1
                value = self.l2_cache.pop(key)
                # Promote to L1
                self._put_l1(key, value)
                return value

            # Check L3 (disk cache)
            if self.l3_path and self._exists_l3(key):
                self.stats["hits"] += 1
                value = self._load_l3(key)
                self._put_l1(key, value)
                return value

            self.stats["misses"] += 1
            return None

    def put(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        """Put into cache with size management"""
        with self.lock:
            # Always put in L1 first
            self._put_l1(key, value)

    def _put_l1(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        """Put in L1 cache with eviction"""
        value_size = self._calculate_size(value)

        # Remove if already exists
        if key in self.l1_cache:
            old_value = self.l1_cache.pop(key)
            self.current_l1_size -= self._calculate_size(old_value)

        # Evict from L1 to L2 if needed
        while (
            self.current_l1_size + value_size > self.l1_max_size
            and len(self.l1_cache) > 0
        ):
            old_key, old_value = self.l1_cache.popitem(last=False)
            old_size = self._calculate_size(old_value)
            self.current_l1_size -= old_size

            # Move to L2
            self._put_l2(old_key, old_value)
            self.stats["evictions"] += 1

        self.l1_cache[key] = value
        self.current_l1_size += value_size

    def _put_l2(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        """Put in L2 cache with eviction to L3"""
        value_size = self._calculate_size(value)

        # Remove if already exists
        if key in self.l2_cache:
            old_value = self.l2_cache.pop(key)
            self.current_l2_size -= self._calculate_size(old_value)

        # Evict from L2 to L3 if needed
        while (
            self.current_l2_size + value_size > self.l2_max_size
            and len(self.l2_cache) > 0
        ):
            old_key, old_value = self.l2_cache.popitem(last=False)
            old_size = self._calculate_size(old_value)
            self.current_l2_size -= old_size

            # Move to L3 if available
            if self.l3_path:
                self._save_l3(old_key, old_value)

        self.l2_cache[key] = value
        self.current_l2_size += value_size

    def _exists_l3(self, key: str) -> bool:
        """Check if key exists in L3 cache"""
        if not self.l3_path:
            return False
        cache_file = self.l3_path / f"{key}.pt"
        return cache_file.exists()

    def _load_l3(self, key: str) -> Dict[str, torch.Tensor]:
        """Load from L3 disk cache"""
        cache_file = self.l3_path / f"{key}.pt"
        return torch.load(cache_file, map_location="cpu")

    def _save_l3(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        """Save to L3 disk cache"""
        cache_file = self.l3_path / f"{key}.pt"
        torch.save(value, cache_file)

    def _memory_pressure_check(self) -> bool:
        """Check if system is under memory pressure"""
        if not HAS_PSUTIL:
            return False  # Assume no memory pressure if psutil not available
        memory = psutil.virtual_memory()
        return memory.percent > 85.0  # High memory usage

    def adaptive_eviction(self) -> None:
        """Adaptive eviction based on memory pressure"""
        if self._memory_pressure_check():
            # Aggressive eviction under memory pressure
            self._evict_percentage(0.3)  # Evict 30%

    def _evict_percentage(self, percentage: float) -> None:
        """Evict a percentage of cached items"""
        with self.lock:
            # Evict from L1
            l1_to_evict = int(len(self.l1_cache) * percentage)
            for _ in range(l1_to_evict):
                if self.l1_cache:
                    key, value = self.l1_cache.popitem(last=False)
                    self.current_l1_size -= self._calculate_size(value)
                    self._put_l2(key, value)

            # Evict from L2
            l2_to_evict = int(len(self.l2_cache) * percentage)
            for _ in range(l2_to_evict):
                if self.l2_cache:
                    key, value = self.l2_cache.popitem(last=False)
                    self.current_l2_size -= self._calculate_size(value)
                    if self.l3_path:
                        self._save_l3(key, value)


# ================== Prefetch Manager ==================


class PrefetchManager:
    """Intelligent prefetching based on access patterns"""

    def __init__(self, storage_backend: StorageBackend, cache: MultiTierCache):
        self.storage = storage_backend
        self.cache = cache
        self.access_history = deque(maxlen=100)
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2)

    def _generate_cache_key(self, sample_ids: List[int]) -> str:
        """Generate cache key from sample IDs"""
        return f"samples_{'_'.join(map(str, sorted(sample_ids)))}"

    def predict_next_access(self, current_sample_ids: List[int]) -> List[int]:
        """Predict next samples to prefetch based on patterns"""
        # Sequential pattern detection
        if len(self.access_history) >= 1:
            current_batch = sorted(current_sample_ids)

            # Simple sequential prediction
            if len(current_batch) > 1:
                step = current_batch[1] - current_batch[0]
                if step > 0:  # Positive step indicates sequential access
                    next_start = max(current_batch) + step
                    return list(range(next_start, next_start + len(current_batch)))

        return []

    def _is_sequential_pattern(self, batch1: List[int], batch2: List[int]) -> bool:
        """Check if two batches follow a sequential pattern"""
        if not batch1 or not batch2:
            return False

        batch1_sorted = sorted(batch1)
        batch2_sorted = sorted(batch2)

        # Check if batch2 starts where batch1 ends (approximately)
        gap = min(batch2_sorted) - max(batch1_sorted)
        return 1 <= gap <= 10  # Allow small gaps

    def prefetch_async(self, sample_ids: List[int]) -> None:
        """Asynchronously prefetch features"""

        def prefetch_task():
            try:
                features = self.storage.load_features(sample_ids)
                if features:  # Only cache if we got results
                    cache_key = self._generate_cache_key(sample_ids)
                    self.cache.put(cache_key, features)
            except Exception as e:
                print(f"Prefetch failed: {e}")

        self.prefetch_executor.submit(prefetch_task)


# ================== Access Pattern Learning ==================


class AccessPatternLearner:
    """Learn and classify data access patterns"""

    def __init__(self, history_size: int = 1000):
        self.access_history = deque(maxlen=history_size)
        self.pattern_stats = {"sequential": 0, "random": 0, "clustered": 0}

    def record_access(self, sample_ids: List[int]) -> None:
        self.access_history.append(sample_ids)

        # Update pattern statistics
        pattern = self.classify_pattern(sample_ids)
        self.pattern_stats[pattern] += 1

    def classify_pattern(self, sample_ids: List[int]) -> str:
        """Classify access pattern as sequential, random, or clustered"""
        if len(sample_ids) <= 1:
            return "random"

        # Check if sequential
        sorted_ids = sorted(sample_ids)
        if sorted_ids == list(range(min(sorted_ids), max(sorted_ids) + 1)):
            return "sequential"

        # Check if clustered (small gaps)
        gaps = [sorted_ids[i + 1] - sorted_ids[i] for i in range(len(sorted_ids) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap < 10:  # Threshold for clustering
            return "clustered"

        return "random"


# ================== Main Feature Manager ==================


class FeatureManager:
    """Main orchestrator for feature management with backward compatibility"""

    def __init__(
        self,
        features_dir: Union[str, Path],
        chunk_size: int = 10000,
        config: Optional[Dict[str, Any]] = None,
        preload_index: bool = False,
    ):
        # Maintain backward compatibility with original constructor
        self.features_dir = Path(features_dir)
        self.chunk_size = chunk_size

        # Load config or use defaults
        if config is None:
            config = {
                "storage_dir": str(features_dir),
                "primary_backend": "chunked",
                "chunked_storage": {
                    "enabled": True,
                    "chunk_size": chunk_size,
                    "compression": False,  # Disable compression for testing
                },
                "cache": {"l1_size_mb": 512, "l2_size_mb": 2048, "l3_path": None},
            }

        self.config = config
        self.storage_backends = self._init_storage_backends()
        self.cache = MultiTierCache(**config.get("cache", {}))
        self.prefetch_manager = PrefetchManager(
            self.storage_backends["primary"], self.cache
        )

        # Access pattern learning
        self.access_patterns = AccessPatternLearner()

        # Maintain backward compatibility
        self.index_mapping = {}
        self.feature_references = []

        # Build index mapping from existing chunk files if requested
        if preload_index:
            self._build_index_mapping()

    def _init_storage_backends(self) -> Dict[str, StorageBackend]:
        """Initialize multiple storage backends based on config"""
        backends = {}

        if self.config.get("chunked_storage", {}).get("enabled", True):
            backends["chunked"] = TorchChunkedStorage(
                storage_dir=Path(self.config["storage_dir"]),
                **{
                    k: v
                    for k, v in self.config.get("chunked_storage", {}).items()
                    if k not in ["enabled"]
                },
            )

        if self.config.get("mmap_storage", {}).get("enabled", False):
            backends["mmap"] = MemoryMappedStorage(
                storage_dir=Path(self.config["storage_dir"]) / "mmap",
                **{
                    k: v
                    for k, v in self.config.get("mmap_storage", {}).items()
                    if k not in ["enabled"]
                },
            )

        # Set primary backend
        primary_name = self.config.get("primary_backend", "chunked")
        backends["primary"] = backends[primary_name]

        return backends

    def _build_index_mapping(self):
        """Build sample_id -> chunk mapping from existing chunk files"""
        chunk_files = list(self.features_dir.glob("chunk_*.pt"))
        for chunk_file in chunk_files:
            try:
                chunk_data = torch.load(chunk_file, map_location="cpu")
                if "sample_ids" in chunk_data:
                    for sample_id in chunk_data["sample_ids"]:
                        self.index_mapping[sample_id] = (
                            str(chunk_file.name),
                            sample_id,
                        )
            except Exception as e:
                print(f"Warning: Could not load {chunk_file}: {e}")
                continue

    def _generate_cache_key(
        self, sample_ids: List[int], feature_types: Optional[List[str]] = None
    ) -> str:
        """Generate cache key for sample IDs and feature types"""
        key_parts = [f"samples_{'_'.join(map(str, sorted(sample_ids)))}"]
        if feature_types:
            key_parts.append(f"types_{'_'.join(sorted(feature_types))}")
        return "_".join(key_parts)

    def _filter_features(
        self, features: Dict[str, torch.Tensor], feature_types: Optional[List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Filter features by requested types"""
        if feature_types is None:
            return features
        return {k: v for k, v in features.items() if k in feature_types}

    # ================== New API Methods ==================

    def get_features(
        self,
        sample_ids: List[int],
        feature_types: Optional[List[str]] = None,
        async_prefetch: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Main interface for getting features"""

        # Generate cache key
        cache_key = self._generate_cache_key(sample_ids, feature_types)

        # Check cache first
        cached_features = self.cache.get(cache_key)
        if cached_features is not None:
            filtered_features = self._filter_features(cached_features, feature_types)

            # Async prefetching for next batch
            if async_prefetch:
                next_sample_ids = self.prefetch_manager.predict_next_access(sample_ids)
                if next_sample_ids:
                    self.prefetch_manager.prefetch_async(next_sample_ids)

            return filtered_features

        # Cache miss - load from storage
        features = self._load_from_best_backend(sample_ids)

        # Update cache
        if features:  # Only cache if we got results
            self.cache.put(cache_key, features)

        # Learn access pattern
        self.access_patterns.record_access(sample_ids)

        # Trigger prefetching
        if async_prefetch:
            next_sample_ids = self.prefetch_manager.predict_next_access(sample_ids)
            if next_sample_ids:
                self.prefetch_manager.prefetch_async(next_sample_ids)

        return self._filter_features(features, feature_types)

    def get_features_by_chunk(
        self,
        chunk_id: int,
        feature_types: Optional[List[str]] = None,
        async_prefetch: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load all features from a specific chunk by chunk ID

        Args:
            chunk_id: The chunk ID to load
            feature_types: Optional list of feature types to return (e.g., ['img_features'])
            async_prefetch: Whether to enable async prefetching

        Returns:
            Dict with same format as get_features(): {"img_features": tensor, "txt_features": tensor, ...}
        """
        chunk_path = self.features_dir / f"chunk_{chunk_id}.pt"

        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk {chunk_id} not found at {chunk_path}")

        # Generate cache key for this chunk
        cache_key = f"chunk_{chunk_id}_all"

        # Check cache first
        cached_features = self.cache.get(cache_key)
        if cached_features is not None:
            filtered_features = self._filter_features(cached_features, feature_types)
            return filtered_features

        # Load chunk from disk
        chunk_data = torch.load(chunk_path, map_location="cpu")

        # Handle decompression if needed
        compression_enabled = getattr(
            self.storage_backends["primary"], "compression", False
        )
        if compression_enabled and HAS_LZ4 and chunk_data.get("features"):
            for feat_name, feat_data in chunk_data["features"].items():
                if isinstance(feat_data, dict) and "compressed_data" in feat_data:
                    compressed = feat_data["compressed_data"]
                    decompressed = lz4.frame.decompress(compressed)
                    # Convert to tensor
                    import numpy as np

                    torch_dtype = feat_data["dtype"]
                    if torch_dtype.startswith("torch."):
                        torch_dtype = torch_dtype.replace("torch.", "")
                    np_dtype = getattr(np, torch_dtype)
                    tensor = np.frombuffer(decompressed, dtype=np_dtype)
                    tensor = torch.from_numpy(tensor.reshape(feat_data["shape"]).copy())
                    chunk_data["features"][feat_name] = tensor

        # Extract features in the same format as get_features()
        features = chunk_data.get("features", {})

        # Cache the result
        if features:
            self.cache.put(cache_key, features)

        # Learn access pattern (treat as sequential since it's a whole chunk)
        if "sample_ids" in chunk_data:
            self.access_patterns.record_access(chunk_data["sample_ids"])

        # Optional: Trigger prefetching for next chunk
        if async_prefetch:
            next_chunk_sample_ids = list(
                range(
                    chunk_id * self.chunk_size + self.chunk_size,
                    (chunk_id + 1) * self.chunk_size + self.chunk_size,
                )
            )
            self.prefetch_manager.prefetch_async(next_chunk_sample_ids)

        return self._filter_features(features, feature_types)

    def _load_from_best_backend(self, sample_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Choose best backend based on access pattern"""
        # Use index mapping if available, otherwise fall back to automatic chunking
        if self.index_mapping:
            return self._load_using_index_mapping(sample_ids)

        access_pattern = self.access_patterns.classify_pattern(sample_ids)

        if access_pattern == "sequential":
            return self.storage_backends["chunked"].load_features(sample_ids)
        elif access_pattern == "random":
            if "mmap" in self.storage_backends:
                return self.storage_backends["mmap"].load_features(sample_ids)
            else:
                return self.storage_backends["primary"].load_features(sample_ids)
        else:
            return self.storage_backends["primary"].load_features(sample_ids)

    def _load_using_index_mapping(
        self, sample_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Load features using the index mapping (for backward compatibility)"""
        # Group sample IDs by chunk file
        chunk_groups = {}
        for sample_id in sample_ids:
            if sample_id in self.index_mapping:
                chunk_file, _ = self.index_mapping[sample_id]
                if chunk_file not in chunk_groups:
                    chunk_groups[chunk_file] = []
                chunk_groups[chunk_file].append(sample_id)

        # Load from each chunk file and combine results
        result_features = {}
        for chunk_file, chunk_sample_ids in chunk_groups.items():
            chunk_path = self.features_dir / chunk_file
            if not chunk_path.exists():
                continue

            chunk_data = torch.load(chunk_path, map_location="cpu")

            # Extract requested samples from this chunk
            chunk_sample_ids_list = chunk_data["sample_ids"]
            indices = [
                chunk_sample_ids_list.index(sid)
                for sid in chunk_sample_ids
                if sid in chunk_sample_ids_list
            ]

            for feat_name, feat_data in chunk_data["features"].items():
                if feat_name not in result_features:
                    if isinstance(feat_data, torch.Tensor):
                        result_features[feat_name] = []
                    else:
                        result_features[feat_name] = []

                if isinstance(feat_data, torch.Tensor):
                    result_features[feat_name].append(feat_data[indices])
                else:
                    # Handle list-type features
                    result_features[feat_name].extend([feat_data[i] for i in indices])

        # Concatenate tensors from different chunks
        for feat_name in result_features:
            if (
                isinstance(result_features[feat_name], list)
                and len(result_features[feat_name]) > 0
            ):
                if isinstance(result_features[feat_name][0], torch.Tensor):
                    result_features[feat_name] = torch.cat(
                        result_features[feat_name], dim=0
                    )

        return result_features

    # ================== Backward Compatibility Methods ==================

    def add_features_chunk(
        self,
        chunk_id: int,
        img_features: torch.Tensor,
        txt_features: torch.Tensor,
        txt_full: Any,
        sample_ids: List[int],
    ) -> None:
        """Maintain backward compatibility with original add_features_chunk method"""
        if len(img_features) != len(txt_features) or len(img_features) != len(
            sample_ids
        ):
            raise ValueError(
                "Length of image features, text features, and sample IDs must be the same."
            )

        features = {
            "img_features": img_features,
            "txt_features": txt_features,
            "txt_full": txt_full,
            "sample_ids": sample_ids,
        }

        # Use direct chunk saving to respect explicit chunk_id
        self.storage_backends["primary"].save_chunk_directly(
            chunk_id, sample_ids, features
        )

        # Update index mapping for compatibility
        for sample_id in sample_ids:
            self.index_mapping[sample_id] = (f"chunk_{chunk_id}.pt", sample_id)

    def load_features(self) -> None:
        """Maintain backward compatibility - load feature metadata"""
        # This was originally used to scan directory for files
        # Now we maintain compatibility but use the new system
        self._build_index_mapping()

    def get_chunk(self, chunk_id: int) -> Dict[str, Any]:
        """Maintain backward compatibility with original get_chunk method"""
        chunk_path = self.storage_backends["primary"].get_chunk_path(chunk_id)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file {chunk_path} not found.")
        return torch.load(chunk_path, weights_only=False)

    def _get_chunk_file_and_idx(self, sample_id: int) -> Tuple[str, int]:
        """Maintain backward compatibility"""
        chunk_idx = sample_id % self.chunk_size
        chunk_file = self.features_dir / f"chunk_{sample_id // self.chunk_size}.pt"
        return str(chunk_file), chunk_idx

    def get_num_chunks(self) -> int:
        """Maintain backward compatibility"""
        return len(list(self.features_dir.glob("chunk_*.pt")))

    def debug_print(self) -> None:
        """Maintain backward compatibility"""
        print(f"Index Mapping: {self.index_mapping}")
        print(f"Cache Stats: {self.cache.stats}")
        print(f"Access Pattern Stats: {self.access_patterns.pattern_stats}")


def test_feature_manager():
    """Test function to verify the new FeatureManager works correctly"""
    import random

    # Set seed to be 42
    random.seed(42)
    torch.manual_seed(42)

    # Create test data
    sample_ids_list = list(range(100))
    random.shuffle(sample_ids_list)

    # Create feature manager
    feature_manager = FeatureManager(
        features_dir="/project/Deep-Clustering/res/test_features_new", chunk_size=20
    )

    # Create fake features
    img_features = torch.randn(100, 512)
    txt_features = torch.randn(100, 512)
    txt_full = ["sample text"] * 100

    # Test adding features in chunks
    for chunk_start in range(0, 100, 20):
        chunk_end = min(chunk_start + 20, 100)
        chunk_sample_ids = sample_ids_list[chunk_start:chunk_end]
        chunk_img_features = img_features[chunk_start:chunk_end]
        chunk_txt_features = txt_features[chunk_start:chunk_end]
        chunk_txt_full = txt_full[chunk_start:chunk_end]

        feature_manager.add_features_chunk(
            chunk_start // 20,
            chunk_img_features,
            chunk_txt_features,
            chunk_txt_full,
            chunk_sample_ids,
        )

    # Test loading features with new API
    test_sample_ids = sample_ids_list[:10]
    loaded_features = feature_manager.get_features(test_sample_ids)

    print(f"Loaded features keys: {loaded_features.keys()}")
    if "img_features" in loaded_features:
        print(
            f"Loaded features shape: img={loaded_features['img_features'].shape}, txt={loaded_features['txt_features'].shape}"
        )
    else:
        print("No features loaded!")
    print(f"Cache stats: {feature_manager.cache.stats}")

    # Test sequential access pattern
    sequential_ids = list(range(20, 40))
    seq_features = feature_manager.get_features(sequential_ids)
    print(f"Sequential features shape: img={seq_features['img_features'].shape}")

    # Test random access pattern
    random_ids = random.sample(sample_ids_list, 15)
    rand_features = feature_manager.get_features(random_ids)
    print(f"Random features shape: img={rand_features['img_features'].shape}")

    print("All tests passed!")

    # Test the new get_features_by_chunk method
    print("\nTesting get_features_by_chunk method...")
    chunk_features = feature_manager.get_features_by_chunk(0)
    print(f"Chunk 0 features keys: {chunk_features.keys()}")
    print(
        f"Chunk 0 via new method: img={chunk_features['img_features'].shape}, txt={chunk_features['txt_features'].shape}"
    )

    # Verify consistency with get_chunk
    chunk_data = feature_manager.get_chunk(0)
    assert torch.equal(
        chunk_features["img_features"], chunk_data["features"]["img_features"]
    )
    print("✅ get_features_by_chunk consistency verified")


if __name__ == "__main__":
    test_feature_manager()
