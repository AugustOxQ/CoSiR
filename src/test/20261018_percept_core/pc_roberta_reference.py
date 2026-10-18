"""Verbatim September extraction helper for the RoBERTa smoke comparison."""
# Source: 1b70d7f:src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py.
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = 'SamLowe/roberta-base-go_emotions'
BATCH_SIZE = 256
MAX_LENGTH = 64


def caption_text(record: dict) -> str:
    """Match ArtELingo's string-or-singleton-list caption convention."""
    caption = record["caption"]
    return caption if isinstance(caption, str) else caption[0]


def extract_affect_embedding_nodes(train_json: str, paintings: list[str], device: str, log) -> np.ndarray:
    """Mean-pool masked RoBERTa token embeddings, then captions by painting."""
    log(f"Loading embedding-level GoEmotions encoder ({MODEL_NAME}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with open(train_json) as train_file:
        train = json.load(train_file)
    painting_to_idx = {painting: index for index, painting in enumerate(paintings)}
    row_node_indices = []
    captions = []
    for record in train:
        if record.get("language", "english").lower() != "english":
            continue
        try:
            row_node_indices.append(painting_to_idx[record["painting"]])
        except KeyError as exc:
            raise RuntimeError(
                "Caption row references a painting absent from deduplicated features: "
                f"{record['painting']}"
            ) from exc
        captions.append(caption_text(record))
    if not captions:
        raise RuntimeError("No English caption rows found in TRAIN_JSON.")

    embedding_sums = None
    embedding_counts = np.zeros(len(paintings), dtype=np.int32)
    total_batches = (len(captions) + BATCH_SIZE - 1) // BATCH_SIZE
    log(
        f"Extracting embeddings for {len(captions):,} English caption rows in "
        f"{total_batches} batches (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH})..."
    )
    with torch.no_grad():
        for start in range(0, len(captions), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(captions))
            encoded = tokenizer(
                captions[start:end], padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            ).to(device)
            outputs = model(**encoded).last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            caption_embeddings = (
                (outputs * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp_min(1)
            ).cpu().numpy().astype(np.float32, copy=False)
            if embedding_sums is None:
                embedding_sums = np.zeros(
                    (len(paintings), caption_embeddings.shape[1]), dtype=np.float32
                )
            node_indices = np.asarray(row_node_indices[start:end], dtype=np.intp)
            np.add.at(embedding_sums, node_indices, caption_embeddings)
            np.add.at(embedding_counts, node_indices, 1)
            if (start // BATCH_SIZE + 1) % 100 == 0 or end == len(captions):
                log(f"Affect encoder progress: {end:,}/{len(captions):,} caption rows.")

    missing = np.flatnonzero(embedding_counts == 0)
    if len(missing):
        raise RuntimeError(f"{len(missing)} deduplicated paintings have no English affect embeddings.")
    return embedding_sums / embedding_counts[:, None]

