"""
Sanity check: can (img_emb - txt_emb) separate caption types?
Uses the Impressions dataset (4 types: aesthetic/caption/description/impression).
Random baseline = 0.25.
"""

import json
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.model import CoSiRModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

ANNOTATION_PATH = "/project/Impressions/metadata/impressions_train.json"
IMAGE_PATH = "/project/Impressions/media"
CLIP_MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
MAX_SAMPLES = 2000  # cap for speed; set None to use all
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TYPE_TO_INT = {"aesthetic": 0, "caption": 1, "description": 2, "impression": 3}


class ImpressionsExtractDataset(Dataset):
    def __init__(self, annotations, image_path, processor):
        self.annotations = annotations
        self.image_path = image_path
        self.processor = processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.image_path, ann["image"])).convert("RGB")
        img_input = self.processor(images=img, return_tensors="pt")
        img_input = {k: v.squeeze(0) for k, v in img_input.items()}
        txt_input = self.processor(
            text=ann["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        txt_input = {k: v.squeeze(0) for k, v in txt_input.items()}
        label = TYPE_TO_INT[ann["caption_type"]]
        return img_input, txt_input, label


def collate_fn(batch):
    imgs, txts, labels = zip(*batch)
    img_batch = {k: torch.stack([d[k] for d in imgs]) for k in imgs[0]}
    txt_batch = {k: torch.stack([d[k] for d in txts]) for k in txts[0]}
    return img_batch, txt_batch, torch.tensor(labels)


def main():
    print(f"Device: {DEVICE}")
    annotations = json.load(open(ANNOTATION_PATH))
    if MAX_SAMPLES:
        annotations = annotations[:MAX_SAMPLES]
    print(f"Samples: {len(annotations)}")

    from collections import Counter
    print("Label distribution:", Counter(a["caption_type"] for a in annotations))

    processor = AutoProcessor.from_pretrained(CLIP_MODEL, use_fast=False)

    model = CoSiRModel(
        backbone_model=CLIP_MODEL,
        label_dim=512,
        num_layers=2,
        d_model=512,
        num_conditions=12,
        dropout=0.0,
    ).to(DEVICE)
    model.eval()

    dataset = ImpressionsExtractDataset(annotations, IMAGE_PATH, processor)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    all_diffs = []
    all_img_embs = []
    all_txt_embs = []
    all_labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for img_batch, txt_batch, labels in loader:
            img_batch = {k: v.to(DEVICE) for k, v in img_batch.items()}
            txt_batch = {k: v.to(DEVICE) for k, v in txt_batch.items()}

            img_emb, txt_emb, _, _ = model.encode_img_txt(img_batch, txt_batch)

            all_diffs.append((img_emb - txt_emb).cpu())
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())
            all_labels.extend(labels.numpy().tolist())

    diff = torch.cat(all_diffs).numpy()
    img_embs = torch.cat(all_img_embs).numpy()
    txt_embs = torch.cat(all_txt_embs).numpy()
    labels = np.array(all_labels)

    print(f"\nEmbedding shape: {diff.shape}")
    print(f"Label counts: {np.bincount(labels)}")

    results = {}

    for name, X in [("diff (img-txt)", diff), ("img only", img_embs), ("txt only", txt_embs)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results[name] = acc
        print(f"  [{name}]  accuracy = {acc:.4f}  (random baseline = 0.25)")

    print("\nSummary:")
    for k, v in results.items():
        gain = v - 0.25
        print(f"  {k:<20s}  acc={v:.4f}  gain over random={gain:+.4f}")


if __name__ == "__main__":
    main()
