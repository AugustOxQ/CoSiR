import os
from PIL import ImageFile
from torch.utils.data import Dataset
from datasets import load_dataset, config

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images


class FeatureExtractionConceptualDataset(Dataset):
    def __init__(
        self, annotation_path: str, image_path: str, processor, ratio=1
    ) -> None:
        config.HF_DATASETS_CACHE = annotation_path

        ratio = int(min(max(ratio, 0.01), 1) * 100)

        # Pre-load length of annotations so we can use streaming
        annotation_length_path = os.path.join(annotation_path, "annotation_length.txt")
        self.annotation_length = -1
        if os.path.exists(annotation_length_path):
            self.annotation_length = int(open(annotation_length_path).read())
            streaming = True
        else:
            streaming = False

        # Load annotations
        self.annotations = load_dataset(
            image_path,
            cache_dir=annotation_path,
            split=f"train[:{ratio}%]",
            streaming=streaming,  # But here we need to get sample ids, so we can't use streaming
        )

        # If we didn't pre-load the length of annotations, we need to load the annotations to get the length
        if self.annotation_length == -1:
            self.annotation_length = len(self.annotations)
            open(annotation_length_path, "w").write(str(self.annotation_length))

        # Assign unique numeric IDs to each sample
        self.sample_ids = {
            i: idx for idx, i in enumerate(range(self.annotation_length))
        }

        self.processor = processor

    def __len__(self):
        return self.annotation_length

    def __getitem__(self, idx: int) -> tuple:
        """Get the processed image and textual annotation for a given index."""
        annotation = self.annotations[idx]
        image_input = self.processor(images=annotation["jpg"], return_tensors="pt")

        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = annotation["txt"]
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
