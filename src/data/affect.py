"""GoEmotions affect probabilities for captions.

The one external training signal the parent spec allows since its 2026-09-30 amendment: the RoBERTa model fine-tuned
on GoEmotions (Reddit comments, 28 emotion categories) that PercepT uses as its affect input. It was never trained on
ArtELingo. It is loaded offline from the local Hugging Face cache.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
GOEMOTIONS_NUM_LABELS = 28


def load_goemotions(model_name: str = GOEMOTIONS_MODEL, device=None):
    """Tokenizer and eval-mode model from the local HF cache (``local_files_only``); raises OSError if not cached."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    if model.config.num_labels != GOEMOTIONS_NUM_LABELS:
        raise ValueError(f"{model_name} has {model.config.num_labels} labels, expected {GOEMOTIONS_NUM_LABELS}")
    return tokenizer, model.to(device or "cpu").eval()


def goemotions_probabilities(texts, device=None, batch_size: int = 256, max_length: int = 64,
                             model_name: str = GOEMOTIONS_MODEL, loaded=None) -> np.ndarray:
    """Sigmoid probabilities over the 28 GoEmotions labels, one float32 row per text, in input order.

    ``loaded`` is a ``load_goemotions`` result to reuse across calls; otherwise the model is loaded here.
    """
    texts = list(texts)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("every text must be a str")
    if not texts:
        return np.zeros((0, GOEMOTIONS_NUM_LABELS), dtype=np.float32)
    tokenizer, model = loaded if loaded is not None else load_goemotions(model_name, device)
    target = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(target)
            out.append(torch.sigmoid(model(**encoded).logits).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)
