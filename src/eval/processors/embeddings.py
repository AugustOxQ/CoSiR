"""Embedding processing utilities."""

from typing import Tuple, List, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from ..config import EvaluationConfig


class EmbeddingProcessor:
    """Handles embedding extraction and preprocessing."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.save_path = ""
        self.use_cache = False

    def set_save_path(self, save_path: str):
        self.save_path = save_path

    def load_embeddings(self):
        file = Path(self.save_path) / "test_cache.pt"
        if file.exists():
            return torch.load(file)
        else:
            raise FileNotFoundError(f"Embeddings file {file} not found")

    def save_embeddings(self, cache: Dict[str, Any]):
        file = Path(self.save_path) / "test_cache.pt"
        torch.save(cache, file)
        print(f"Embeddings saved to {file}")

    def extract_embeddings(self, model, processor, dataloader: DataLoader):
        """Extract embeddings from model for evaluation dataset.

        Returns:
            Tuple of (image_embeddings, text_embeddings, text_full, text_to_image_map, image_to_text_map)
        """

        if self.use_cache:
            cache = self.load_embeddings()
            print("Using cached embeddings")
            return (
                cache["all_img_emb"],
                cache["all_txt_emb"],
                cache["all_img_full"],
                None,  # cache["all_txt_full"],
                cache["all_raw_text"],
                cache["text_to_image_map"],
                cache["image_to_text_map"],
            )

        cache = {}

        all_img_emb = []
        all_txt_emb = []
        all_img_full = []
        # all_txt_full = []
        all_raw_text = []
        image_to_text_map = []
        text_to_image_map = []

        text_index = 0
        image_index = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                image, raw_text = batch
                image_input = image.to(self.config.device)
                batch_size = image_input["pixel_values"].size(0)

                captions_per_image = getattr(
                    dataloader.dataset, "captions_per_image", 5
                )

                # Flatten and process text
                raw_text_list = self._flatten_text(
                    raw_text, batch_size, captions_per_image
                )

                all_raw_text.extend(raw_text_list)

                # Tokenize text
                text_input = processor(
                    text=raw_text_list,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.max_text_length,
                ).to(self.config.device)

                # Update mappings
                for _ in range(batch_size):
                    text_indices = list(
                        range(text_index, text_index + captions_per_image)
                    )
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                # Extract embeddings
                img_emb, txt_emb, img_full, _ = model.encode_img_txt(
                    image_input, text_input
                )

                all_img_emb.append(img_emb.cpu())
                all_txt_emb.append(txt_emb.cpu())
                # all_txt_full.append(txt_full.cpu())
                all_img_full.append(img_full.cpu())

                # Cleanup
                del img_emb, txt_emb, img_full, image_input, text_input
                torch.cuda.empty_cache()

        # Concatenate all embeddings
        all_img_emb = torch.cat(all_img_emb, dim=0)
        all_txt_emb = torch.cat(all_txt_emb, dim=0)
        # all_txt_full = torch.cat(all_txt_full, dim=0)
        all_img_full = torch.cat(all_img_full, dim=0)

        # Convert mappings to tensors
        text_to_image_map = torch.LongTensor(text_to_image_map).to(self.config.device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(self.config.device)

        cache["all_img_emb"] = all_img_emb
        cache["all_txt_emb"] = all_txt_emb
        # cache["all_txt_full"] = all_txt_full
        cache["all_img_full"] = all_img_full
        cache["all_raw_text"] = all_raw_text
        cache["text_to_image_map"] = text_to_image_map
        cache["image_to_text_map"] = image_to_text_map

        if self.save_path != "":
            self.save_embeddings(cache)
            self.use_cache = True

        return (
            cache["all_img_emb"],
            cache["all_txt_emb"],
            cache["all_img_full"],
            None,  # cache["all_txt_full"],
            cache["all_raw_text"],
            cache["text_to_image_map"],
            cache["image_to_text_map"],
        )

    def _flatten_text(
        self, raw_text: Any, batch_size: int, captions_per_image: int
    ) -> List[str]:
        """Flatten text input based on dataset structure."""
        raw_text_list = []

        for b in range(batch_size):
            if captions_per_image == 1:
                raw_text_list.append(raw_text[b])
            else:
                for i in range(captions_per_image):
                    raw_text_list.append(raw_text[i][b])

        return raw_text_list

    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit length."""
        return F.normalize(embeddings, p=2, dim=1)

    def compute_similarity_matrix(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity matrix between two sets of embeddings."""
        # Ensure embeddings are normalized
        embeddings1 = self.normalize_embeddings(embeddings1)
        embeddings2 = self.normalize_embeddings(embeddings2)

        # Compute similarity
        similarity = embeddings1 @ embeddings2.T

        return similarity
