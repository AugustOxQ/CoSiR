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
        self.annotations = load_dataset(
            image_path,
            cache_dir=annotation_path,
            split=f"train[:{ratio}%]",
            # streaming=True, # But here we need to get sample ids, so we can't use streaming
        )

        self.processor = processor

        # Assign unique numeric IDs to each sample
        self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}

    def __len__(self):
        return len(self.annotations)

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
