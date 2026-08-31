import json
import os

import numpy as np
import torch
from datasets import config
from PIL import Image, ImageFile
from torch.utils.data import Dataset, IterableDataset

from typing import List, Optional, Union, Dict

from src.utils import FeatureManager

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images

custom_download_path = "/data/SSD2/HF_datasets"
config.HF_DATASETS_CACHE = custom_download_path


# ── Extraction datasets ────────────────────────────────────────────────────────


class FeatureExtractionDataset(Dataset):
    def __init__(
        self, annotation_path: str, image_path: str, processor, ratio=0.1
    ) -> None:
        """A dataset that is used to pre-extract features from image-text data."""
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor

        # Assign unique numeric IDs to each sample
        self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        """Get the processed image and textual annotation for a given index."""
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.processor(images=raw_image, return_tensors="pt")

        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) is str
            else self.annotations[idx]["caption"][0]
        )

        text_input = self.processor(
            text=raw_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Squeeze batch dimension to match per-sample dict expected by model.encode_txt
        text_input = {
            k: v.squeeze(0) if hasattr(v, "squeeze") else v
            for k, v in text_input.items()
        }

        sample_id = self.sample_ids[idx]

        return image_input, text_input, sample_id


# ── Training datasets (new shard-based API) ───────────────────────────────────


class CoSiRShardDataset(Dataset):
    """
    Training dataset for when CLS features fit in available RAM.

    Loads all shards into RAM at construction time.  DataLoader(shuffle=True)
    then gives true per-epoch random batches with zero disk I/O during training.

    Each __getitem__ returns a dict:
        img_features : float32 [D]
        txt_features : float32 [D]
        sample_ids   : int64   scalar
        img_full     : float32 [...] (only if stored in the feature store)
        txt_full     : float32 [...] (only if stored in the feature store)
    """

    def __init__(
        self,
        feature_manager: FeatureManager,
        feature_types: Optional[List[str]] = None,
    ):
        self.fm = feature_manager
        if feature_types is None:
            feature_types = feature_manager.available_features

        print(
            f"[CoSiRShardDataset] Loading {feature_manager.total_samples:,} samples "
            f"({feature_manager.cls_features_size_gb():.1f} GiB CLS) into RAM…"
        )
        self._data = feature_manager.load_all_to_ram(feature_types)
        print("[CoSiRShardDataset] Done.")

    def __len__(self) -> int:
        return len(self._data["sample_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self._data.items()}


class CoSiRShardStreamDataset(IterableDataset):
    """
    Training dataset for when CLS features do NOT fit in available RAM.

    Streams shards from disk in a randomly shuffled order each epoch, with a
    configurable in-memory shuffle window for within-window randomness.

    Usage:
        dataset = CoSiRShardStreamDataset(feature_manager, window_shards=4)
        loader  = DataLoader(dataset, batch_size=512)
        for epoch in range(epochs):
            dataset.set_epoch(epoch)   # ← call before each epoch
            for batch in loader:
                ...

    With num_workers > 0, shards are partitioned among workers so every sample
    is yielded exactly once per epoch.

    Each yielded item has the same keys as CoSiRShardDataset.
    """

    def __init__(
        self,
        feature_manager: FeatureManager,
        feature_types: Optional[List[str]] = None,
        window_shards: int = 4,
        seed: int = 42,
    ):
        self.fm = feature_manager
        self.feature_types = (
            feature_types
            if feature_types is not None
            else feature_manager.available_features
        )
        self.window_shards = window_shards
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to get a fresh shuffle order."""
        self._epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        rng = np.random.default_rng(self.seed + self._epoch)

        # Shuffle shard order, then partition interleaved across workers
        all_shards = rng.permutation(self.fm.num_shards).tolist()
        worker_shards = all_shards[worker_id::num_workers]

        # Yield samples window by window
        for w_start in range(0, len(worker_shards), self.window_shards):
            window_ids = worker_shards[w_start : w_start + self.window_shards]

            # Load window into RAM
            accum: Dict[str, List[torch.Tensor]] = {}
            for sid in window_ids:
                shard = self.fm.load_shard_to_ram(sid, self.feature_types)
                for k, v in shard.items():
                    accum.setdefault(k, []).append(v)

            window: Dict[str, torch.Tensor] = {
                k: torch.cat(vs, dim=0) for k, vs in accum.items()
            }
            n = len(window["sample_ids"])

            # Shuffle within window
            perm = rng.permutation(n)
            for idx in perm:
                yield {k: v[idx] for k, v in window.items()}


# ── Legacy training datasets (kept for backward compatibility) ─────────────────


class CoSiRTrainingDataset(Dataset):
    """Legacy dataset — use CoSiRShardDataset for new code."""

    def __init__(
        self,
        feature_manager: FeatureManager,
        sample_ids: List[int],
        batch_size: Optional[int] = None,
        enable_prefetch: bool = True,
    ):
        self.feature_manager = feature_manager
        self.sample_ids = sample_ids
        self.batch_size = batch_size
        self.enable_prefetch = enable_prefetch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: Union[int, slice]) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            sample_ids = self.sample_ids[idx]
        else:
            sample_ids = [self.sample_ids[idx]]
        return self.feature_manager.get_features_by_chunk(
            sample_ids[0] // self.feature_manager.chunk_size
        )


class CoSiRTrainingChunkDataset(Dataset):
    """Legacy dataset — use CoSiRShardDataset for new code."""

    def __init__(
        self,
        feature_manager: FeatureManager,
        sample_ids: List[int],
        enable_prefetch: bool = True,
    ):
        self.feature_manager = feature_manager
        self.sample_ids = sample_ids
        self.enable_prefetch = enable_prefetch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> int:
        return 1


# ── Validation dataset ─────────────────────────────────────────────────────────


class CoSiRValidationDataset(Dataset):

    def __init__(
        self,
        annotation_path: str,
        image_path: str,
        processor,
        ratio: float = 0.1,
        crop_num: int = 5,
    ):
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor
        self.crop_num = crop_num
        self.caption_length_list = []
        for i in range(len(self.annotations)):
            if type(self.annotations[i]["caption"]) is str:
                self.caption_length_list.append(1)
            else:
                self.caption_length_list.append(len(self.annotations[i]["caption"]))
        self.captions_per_image = (
            1
            if type(self.annotations[0]["caption"]) is str
            else len(self.annotations[0]["caption"])
        )
        self.captions_per_image = min(self.captions_per_image, self.crop_num)
        print(f"Captions per image: {self.captions_per_image}")

    def get_all_raw_image(self, idx: int = -1):
        self.all_raw_image = []
        for i in range(len(self.annotations)):
            img_path = os.path.join(self.image_path, self.annotations[i]["image"])
            raw_image = Image.open(img_path).convert("RGB")
            self.all_raw_image.append(raw_image)
        if idx == -1:
            return self.all_raw_image
        else:
            return self.all_raw_image[idx]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get the processed image and textual annotation for a given index."""
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.processor(images=raw_image, return_tensors="pt")
        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) is str
            else self.annotations[idx]["caption"][: self.crop_num]
        )

        return image_input, raw_text
