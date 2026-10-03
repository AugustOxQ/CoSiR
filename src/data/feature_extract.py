"""Frozen-backbone feature extraction: CLIP ViT-B/32 and Qwen3-VL-Embedding-2B (CVPR plan Task 6).

Ported from src/test/20261025_backbone_check/extract.py (HFClipLike, QwenEmb). Both encoders return float32,
L2-normalised arrays. PIL images and strings in, numpy out; no file paths.
"""

import os
import sys

_NVIDIA = "/root/miniconda3/envs/CoSiR/lib/python3.10/site-packages/nvidia/cu13/lib"
if _NVIDIA not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{_NVIDIA}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch

QWEN_INSTRUCTION_DEFAULT = "Represent the user's input."
QWEN_MAX_PIXELS = 512 * 32 * 32     # caps image tokens at 512 (backbone check setting)
CLIP_REPO = "openai/clip-vit-base-patch32"
QWEN_REPO = "Qwen/Qwen3-VL-Embedding-2B"


def _l2(x) -> np.ndarray:
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()


def _batches(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


# Image pre-resize copied from the official pipeline: qwen_vl_utils/vision_process.py (smart_resize, to_rgb and the
# resize step of fetch_image), as called by the model snapshot's scripts/qwen3_vl_embedding.py. The official embedder
# resizes with PIL (default resample) to multiples of 32 and then calls the processor with do_resize=False.
_QWEN_FACTOR = 32                      # image_patch_size 16 * spatial merge 2
_QWEN_MIN_PIXELS = 4 * 32 * 32         # the official embedder's MIN_PIXELS
_QWEN_MAX_RATIO = 200


def _qwen_smart_resize(height, width, factor, min_pixels, max_pixels):
    import math
    if max(height, width) / min(height, width) > _QWEN_MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {_QWEN_MAX_RATIO}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _qwen_prepare_image(im, max_pixels):
    from PIL import Image
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    w, h = im.size
    h2, w2 = _qwen_smart_resize(h, w, _QWEN_FACTOR, _QWEN_MIN_PIXELS, max_pixels)
    return im.resize((w2, h2))


class Encoder:
    name: str
    dim: int

    def encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts, batch_size: int = 256) -> np.ndarray:
        raise NotImplementedError


class ClipB32(Encoder):
    name = "clip_b32"
    dim = 512

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.m = AutoModel.from_pretrained(CLIP_REPO, dtype=self.dtype).to(device).eval()
        self.p = AutoProcessor.from_pretrained(CLIP_REPO)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        out = []
        for b in _batches(list(images), batch_size):
            x = self.p.image_processor([im.convert("RGB") for im in b], return_tensors="pt")["pixel_values"]
            f = self.m.get_image_features(pixel_values=x.to(self.device, self.dtype))
            out.append(_l2(getattr(f, "pooler_output", f)))
        return np.concatenate(out)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size: int = 256) -> np.ndarray:
        out = []
        for b in _batches(list(texts), batch_size):
            t = self.p.tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            f = self.m.get_text_features(**t)
            out.append(_l2(getattr(f, "pooler_output", f)))
        return np.concatenate(out)


class Qwen3VLEmb(Encoder):
    name = "qwen3vl_emb_2b"
    dim = 2048

    def __init__(self, device: str = "cuda", instruction: str = QWEN_INSTRUCTION_DEFAULT,
                 max_pixels: int = QWEN_MAX_PIXELS):
        from transformers import AutoModel, Qwen3VLProcessor
        self.device, self.instruction, self.max_pixels = device, instruction, max_pixels
        self.m = AutoModel.from_pretrained(QWEN_REPO, dtype=torch.bfloat16).to(device).eval()
        self.p = Qwen3VLProcessor.from_pretrained(QWEN_REPO, padding_side="right")
        self.p.image_processor.size = {"shortest_edge": 4096, "longest_edge": max_pixels}
        self.p.image_processor.min_pixels, self.p.image_processor.max_pixels = 4096, max_pixels

    def _prompt(self, content):
        conv = [{"role": "system", "content": [{"type": "text", "text": self.instruction}]},
                {"role": "user", "content": content}]
        return self.p.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def _pool(self, ids, mask, pix=None, thw=None, mm=None):
        kw = dict(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        if pix is not None:
            kw.update(pixel_values=pix.to(self.device, torch.bfloat16), image_grid_thw=thw.to(self.device),
                      mm_token_type_ids=mm.to(self.device))
        h = self.m(**kw).last_hidden_state
        last = mask.shape[1] - 1 - mask.flip(1).argmax(1)
        return _l2(h[torch.arange(h.shape[0]), last.to(self.device)])

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 16) -> np.ndarray:
        prompt = self._prompt([{"type": "image"}])
        pad = self.p.tokenizer.pad_token_id
        out = []
        for b in _batches(list(images), batch_size):
            items = [self.p(text=[prompt], images=[_qwen_prepare_image(im, self.max_pixels)], do_resize=False,
                           return_tensors="pt") for im in b]
            L = max(o["input_ids"].shape[1] for o in items)
            ids = torch.full((len(b), L), pad)
            mask = torch.zeros(len(b), L, dtype=torch.long)
            mm = torch.zeros(len(b), L, dtype=torch.long)
            for j, o in enumerate(items):
                n = o["input_ids"].shape[1]
                ids[j, :n], mask[j, :n], mm[j, :n] = o["input_ids"][0], 1, o["mm_token_type_ids"][0]
            out.append(self._pool(ids, mask, torch.cat([o["pixel_values"] for o in items]),
                                  torch.cat([o["image_grid_thw"] for o in items]), mm))
        return np.concatenate(out)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size: int = 128) -> np.ndarray:
        texts = list(texts)
        order = np.argsort([len(s) for s in texts])
        out = np.zeros((len(texts), self.dim), np.float32)
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            t = self.p.tokenizer([self._prompt([{"type": "text", "text": texts[j]}]) for j in idx],
                                 return_tensors="pt", padding=True, truncation=True, max_length=512)
            out[idx] = self._pool(t["input_ids"], t["attention_mask"])
        return out


def load_encoder(name: str, device: str = "cuda", instruction: str = QWEN_INSTRUCTION_DEFAULT) -> Encoder:
    if name == "clip_b32":
        return ClipB32(device)
    if name == "qwen3vl_emb_2b":
        return Qwen3VLEmb(device, instruction)
    raise KeyError(name)
