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
        """
        Initialize the FeatureExtractionDataset class.

        Parameters
        ----------
        annotation_path : str
            Path to the annotation file, expected to be a JSON file.
        image_path : str
            Path to the directory containing the images.
        processor : object
            A processor object for processing the images.
        ratio : float, optional
            The ratio of samples to use from the annotation file, by default 0.1.

        Attributes
        ----------
        annotations : list
            A list of annotations loaded from the JSON file, reduced by the given ratio.
        image_path : str
            The path to the image directory.
        processor : object
            The processor object for processing the images.
        sample_ids : dict
            A dictionary assigning unique numeric IDs to each sample in the annotations list.
        """

        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor

        # Assign unique numeric IDs to each sample
        self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve the processed image and textual annotation for a given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - image_input: The processed image tensor.
            - raw_text: The caption or text associated with the image.
            - sample_id: The unique numeric ID assigned to the sample.
        """

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

        sample_id = self.sample_ids[idx]

        return image_input, raw_text, sample_id


class OptimizedFeatureDataset:
    """Optimized Dataset that leverages the new FeatureManager"""

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

        # Pre-warm cache with first batch
        if len(sample_ids) > 0:
            first_batch = sample_ids[: min(100, len(sample_ids))]
            self.feature_manager.get_features(first_batch)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: Union[int, slice]) -> Dict[str, torch.Tensor]:
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
