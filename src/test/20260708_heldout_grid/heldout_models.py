"""
Held-out encoder registry for the buddy generalization grid.

Six encoders, none of which built the buddy graph (graph = CLIP ViT-B/32, both
towers). Three embed IMAGES, three embed CAPTIONS. Each exposes exactly one of
`encode_images` / `encode_texts` (matching its modality) returning an L2-normalized
float32 numpy array of shape (n, d). All per-model differences (pooling, prefix,
loader) live in the MODELS registry so the extractor stays model-agnostic.

Loaded via plain transformers (no sentence-transformers dependency).

Design: docs/superpowers/specs/2026-07-08-heldout-grid-design.md
"""
import numpy as np
import torch
import torch.nn.functional as F


# ── pooling helpers ──────────────────────────────────────────────────────────

def _cls(out, mask=None):
    """First token of the last hidden state (CLS / register token)."""
    return out.last_hidden_state[:, 0]


def _mean(out, mask):
    """Attention-masked mean over tokens."""
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


# ── registry ─────────────────────────────────────────────────────────────────
# modality: "image" | "text"
# For image models: loader builds (model, processor); embed via `image_fn`.
# For text models:  loader builds (model, tokenizer); pool + optional prefix.

MODELS = {
    # vision
    "dinov2": dict(
        modality="image", hf="facebook/dinov2-small",
        image_kind="backbone", pool=_cls, dim=384,
        note="self-supervised; CLS token",
    ),
    "siglip_v": dict(
        modality="image", hf="google/siglip-base-patch16-224",
        image_kind="get_image_features", dim=768,
        note="sigmoid VLM; pooled image features",
    ),
    "vit_sup": dict(
        modality="image", hf="google/vit-base-patch16-224",
        image_kind="backbone", pool=_cls, dim=768,
        note="ImageNet supervised; CLS token",
    ),
    # language
    "minilm": dict(
        modality="text", hf="sentence-transformers/all-MiniLM-L6-v2",
        pool=_mean, prefix="", dim=384,
        note="mean pooling",
    ),
    "bge": dict(
        modality="text", hf="BAAI/bge-base-en-v1.5",
        pool=_cls, prefix="", dim=768,
        note="CLS pooling",
    ),
    "e5": dict(
        modality="text", hf="intfloat/e5-base-v2",
        pool=_mean, prefix="query: ", dim=768,
        note="mean pooling; symmetric 'query: ' prefix",
    ),
}


class HeldoutEncoder:
    """Uniform batched encoder for one registry entry."""

    def __init__(self, key: str, device: str | None = None):
        if key not in MODELS:
            raise KeyError(f"unknown model '{key}'; known: {list(MODELS)}")
        self.key = key
        self.cfg = MODELS[key]
        self.modality = self.cfg["modality"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        hf = self.cfg["hf"]
        if self.modality == "image":
            self.processor = AutoProcessor.from_pretrained(hf)
            self.model = AutoModel.from_pretrained(hf).to(self.device).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf)
            self.model = AutoModel.from_pretrained(hf).to(self.device).eval()

    @property
    def dim(self) -> int:
        return self.cfg["dim"]

    # ── image path ──
    @torch.no_grad()
    def encode_images(self, images) -> np.ndarray:
        if self.modality != "image":
            raise TypeError(f"{self.key} is a text model")
        inp = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.cfg["image_kind"] == "get_image_features":
            feats = self.model.get_image_features(**inp)
            # transformers 5.x may return a pooling-output object, not a tensor
            if not torch.is_tensor(feats):
                feats = feats.pooler_output
        else:  # backbone: pool the last hidden state
            out = self.model(**inp)
            feats = self.cfg["pool"](out)
        return self._norm(feats)

    # ── text path ──
    @torch.no_grad()
    def encode_texts(self, texts) -> np.ndarray:
        if self.modality != "text":
            raise TypeError(f"{self.key} is an image model")
        prefix = self.cfg.get("prefix", "")
        if prefix:
            texts = [prefix + t for t in texts]
        inp = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=256, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        feats = self.cfg["pool"](out, inp["attention_mask"])
        return self._norm(feats)

    @staticmethod
    def _norm(feats: torch.Tensor) -> np.ndarray:
        return F.normalize(feats.float(), dim=1).cpu().numpy().astype(np.float32)
