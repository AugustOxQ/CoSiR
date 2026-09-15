"""CLIP-text-anchor semantic-direction construction (StyleCLIP/ActAdd-style
difference-of-embeddings) plus matched random-prompt control directions, for
the Exp. 17.1 condition-space audit. Uses the raw HF CLIP text tower only —
never `CoSiRModel` (see this plan's Global Constraints).
"""
import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BACKBONE = "openai/clip-vit-base-patch32"

# Fixed neutral word pool for control (meaningless) prompt-pair directions —
# unrelated to any of this audit's target axes, sampled deterministically.
CONTROL_WORD_POOL = [
    "table", "cloud", "river", "engine", "pencil", "mountain", "bottle", "ladder",
    "window", "carpet", "hammer", "bicycle", "lantern", "curtain", "basket", "kettle",
    "anchor", "blanket", "compass", "drum", "shovel", "mirror", "ribbon", "pillow",
    "faucet", "trumpet", "sandal", "wrench", "candle", "barrel",
]
CONTROL_TEMPLATES = ["a photo of a {w}", "an image of a {w}", "a picture of a {w}"]


def direction_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    direction = emb_a.mean(axis=0) - emb_b.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("degenerate direction (pole A and pole B means are identical)")
    return direction / norm


def load_clip_text_tower(device="cpu"):
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    return model, tokenizer


def encode_prompts(model, tokenizer, prompts, device="cpu"):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
    return emb.cpu().numpy().astype(np.float32)


def build_direction(model, tokenizer, prompts_a, prompts_b, device="cpu"):
    emb_a = encode_prompts(model, tokenizer, prompts_a, device)
    emb_b = encode_prompts(model, tokenizer, prompts_b, device)
    return direction_from_embeddings(emb_a, emb_b)


def build_control_directions(model, tokenizer, n=20, seed=42, device="cpu"):
    rng = random.Random(seed)
    directions = []
    for _ in range(n):
        w1, w2 = rng.sample(CONTROL_WORD_POOL, 2)
        d = build_direction(
            model, tokenizer,
            [template.format(w=w1) for template in CONTROL_TEMPLATES],
            [template.format(w=w2) for template in CONTROL_TEMPLATES],
            device=device,
        )
        directions.append(d)
    return directions
