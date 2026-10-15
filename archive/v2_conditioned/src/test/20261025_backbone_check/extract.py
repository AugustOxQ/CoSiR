"""Frozen-backbone feature extraction for the backbone check (throwaway, diagnostic).

Each backbone exposes `.images(paths) -> (n,d) float32 L2-normalised` and `.texts(strs) -> (n,d)`.
Own preprocessing, projected shared embedding, fp16/bf16 autocast, batched inference.
"""
import sys
sys.path.insert(0, "/data/SSD2/pyenvs/backbone_check")  # open_clip / timm live here only
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEV = "cuda"
NW = 8


def _l2(x):
    x = x.float()
    return torch.nn.functional.normalize(x, dim=-1).cpu().numpy()


class _ImgDS(Dataset):
    def __init__(self, paths, fn): self.paths, self.fn = paths, fn
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.fn(Image.open(self.paths[i]).convert("RGB"))


def _loader(ds, bs, collate=None):
    return DataLoader(ds, batch_size=bs, num_workers=NW, shuffle=False, collate_fn=collate, prefetch_factor=4)


def _batches(xs, bs):
    for i in range(0, len(xs), bs): yield xs[i:i + bs]


class HFClipLike:
    """CLIP ViT-B/32 and SigLIP 2 through transformers get_*_features (projection outputs)."""
    def __init__(self, repo, text_kw, dtype=torch.float16, lower=False):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(repo, dtype=dtype).to(DEV).eval()
        self.p = AutoProcessor.from_pretrained(repo)
        self.text_kw, self.lower, self.dtype = text_kw, lower, dtype

    @torch.no_grad()
    def images(self, paths, bs=128):
        ip = self.p.image_processor
        ds = _ImgDS(paths, lambda im: ip(im, return_tensors="pt")["pixel_values"][0])
        out = []
        for x in _loader(ds, bs):
            f = self.m.get_image_features(pixel_values=x.to(DEV, self.dtype))
            out.append(_l2(getattr(f, "pooler_output", f)))
        return np.concatenate(out)

    @torch.no_grad()
    def texts(self, strs, bs=512):
        out = []
        for b in _batches(list(strs), bs):
            if self.lower: b = [s.lower() for s in b]
            t = self.p.tokenizer(b, return_tensors="pt", **self.text_kw).to(DEV)
            f = self.m.get_text_features(**t)
            out.append(_l2(getattr(f, "pooler_output", f)))
        return np.concatenate(out)


class PECore:
    def __init__(self, repo="hf-hub:timm/PE-Core-L-14-336"):
        import open_clip
        self.m, _, self.tf = open_clip.create_model_and_transforms(repo)
        self.m = self.m.to(DEV).half().eval()
        self.tok = open_clip.get_tokenizer(repo)

    @torch.no_grad()
    def images(self, paths, bs=128):
        out = []
        for x in _loader(_ImgDS(paths, self.tf), bs):
            out.append(_l2(self.m.encode_image(x.to(DEV).half())))
        return np.concatenate(out)

    @torch.no_grad()
    def texts(self, strs, bs=512):
        out = []
        for b in _batches(list(strs), bs):
            out.append(_l2(self.m.encode_text(self.tok(b).to(DEV))))
        return np.concatenate(out)


class QwenEmb:
    INSTR = "Represent the user's input."  # model default instruction
    MAX_PIXELS = 512 * 32 * 32            # cap image tokens at 512 (default 1.3M px is far slower; kept same for all rows)

    def __init__(self, repo="Qwen/Qwen3-VL-Embedding-2B"):
        from transformers import AutoModel, Qwen3VLProcessor
        self.m = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to(DEV).eval()
        self.p = Qwen3VLProcessor.from_pretrained(repo, padding_side="right")
        self.p.image_processor.size = {"shortest_edge": 4096, "longest_edge": self.MAX_PIXELS}
        self.p.image_processor.min_pixels, self.p.image_processor.max_pixels = 4096, self.MAX_PIXELS

    def _prompt(self, content):
        conv = [{"role": "system", "content": [{"type": "text", "text": self.INSTR}]},
                {"role": "user", "content": content}]
        return self.p.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def _pool(self, ids, mask, pix=None, thw=None, mm=None):
        kw = dict(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
        if pix is not None: kw.update(pixel_values=pix.to(DEV, torch.bfloat16), image_grid_thw=thw.to(DEV), mm_token_type_ids=mm.to(DEV))
        h = self.m(**kw).last_hidden_state
        last = mask.shape[1] - 1 - mask.flip(1).argmax(1)
        return _l2(h[torch.arange(h.shape[0]), last.to(DEV)])

    @torch.no_grad()
    def images(self, paths, bs=32):
        prompt = self._prompt([{"type": "image"}])
        pad = self.p.tokenizer.pad_token_id

        class DS(Dataset):
            def __len__(s): return len(paths)
            def __getitem__(s, i):
                o = self.p(text=[prompt], images=[Image.open(paths[i]).convert("RGB")], return_tensors="pt")
                return o["input_ids"][0], o["pixel_values"], o["image_grid_thw"], o["mm_token_type_ids"][0]

        def collate(b):
            L = max(len(x[0]) for x in b)
            ids = torch.full((len(b), L), pad); mask = torch.zeros(len(b), L, dtype=torch.long); mm = torch.zeros(len(b), L, dtype=torch.long)
            for j, x in enumerate(b): ids[j, :len(x[0])] = x[0]; mask[j, :len(x[0])] = 1; mm[j, :len(x[0])] = x[3]
            return ids, mask, torch.cat([x[1] for x in b]), torch.cat([x[2] for x in b]), mm

        out = []
        for ids, mask, pix, thw, mm in _loader(DS(), bs, collate):
            out.append(self._pool(ids, mask, pix, thw, mm))
        return np.concatenate(out)

    @torch.no_grad()
    def texts(self, strs, bs=128):
        strs = list(strs); order = np.argsort([len(s) for s in strs]); out = np.zeros((len(strs), 2048), np.float32)
        for i in range(0, len(strs), bs):
            idx = order[i:i + bs]
            t = self.p.tokenizer([self._prompt([{"type": "text", "text": strs[j]}]) for j in idx],
                                 return_tensors="pt", padding=True, truncation=True, max_length=512)
            out[idx] = self._pool(t["input_ids"], t["attention_mask"])
        return out


def build(name):
    if name == "clip":
        return HFClipLike("openai/clip-vit-base-patch32", dict(padding=True, truncation=True, max_length=77), torch.float16)
    if name == "siglip2":
        return HFClipLike("google/siglip2-so400m-patch14-384",
                          dict(padding="max_length", truncation=True, max_length=64), torch.bfloat16, lower=True)
    if name == "pe":
        return PECore()
    if name == "qwen":
        return QwenEmb()
    raise KeyError(name)
