import json
import os

import torch
from datasets import config
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from typing import List, Optional, Union, Dict

from src.utils import FeatureManager

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images

custom_download_path = "/data/SSD2/HF_datasets"
config.HF_DATASETS_CACHE = custom_download_path


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
            max_length=77,
        )

        # Squeeze batch dimension to match per-sample dict expected by model.encode_txt
        text_input = {
            k: v.squeeze(0) if hasattr(v, "squeeze") else v
            for k, v in text_input.items()
        }

        sample_id = self.sample_ids[idx]

        return image_input, text_input, sample_id


class CoSiRTrainingDataset(Dataset):
    """A dataset that is used to train the CoSiR model."""

    def __init__(
        self,
        feature_manager: FeatureManager,
        sample_ids: List[int],
        batch_size: Optional[int] = None,
        enable_prefetch: bool = True,
    ):
        """A dataset that is used to train the CoSiR model."""
        self.feature_manager = feature_manager
        self.sample_ids = sample_ids  # list of sample ids
        self.batch_size = batch_size
        self.enable_prefetch = enable_prefetch  # whether to prefetch features

        # Pre-warm cache with first batch
        if len(sample_ids) > 0:
            first_batch = sample_ids[: min(100, len(sample_ids))]
            self.feature_manager.get_features(first_batch)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Dict[
        str, torch.Tensor
    ]:  # TODO: Figure out whether I need natural idx or sample ids as input
        """Get the features for a given index."""
        if isinstance(idx, slice):
            # Batch access
            sample_ids = self.sample_ids[idx]
            return self.feature_manager.get_features(
                sample_ids, async_prefetch=self.enable_prefetch
            )
        else:
            # Single sample access
            sample_id = self.sample_ids[idx]
            return self.feature_manager.get_features(
                [sample_id], async_prefetch=self.enable_prefetch
            )


class CoSiRTrainingChunkDataset(Dataset):
    """A dataset that is used to train the CoSiR model."""

    def __init__(
        self,
        feature_manager: FeatureManager,
        sample_ids: List[int],
        enable_prefetch: bool = True,
    ):
        """A dataset that is used to train the CoSiR model."""
        self.feature_manager = feature_manager
        self.sample_ids = sample_ids  # list of sample ids
        self.enable_prefetch = enable_prefetch  # whether to prefetch features

        # Pre-warm cache with first batch
        if len(sample_ids) > 0:
            first_batch = sample_ids[: min(100, len(sample_ids))]
            self.feature_manager.get_features(first_batch)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> int:
        """Only use for getting chunk_id"""
        x = 1
        return x


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
