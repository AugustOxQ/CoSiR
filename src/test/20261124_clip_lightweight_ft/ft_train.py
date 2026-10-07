"""Trainer for the lightweight CLIP fine-tuning comparator (spec 2026-10-07 §3-§4): LP, LB and LoRA.

  python ft_train.py --variant {LP,LB,LoRA} --lr <float> --epochs 10 --out <dir> [--smoke] [--device cuda|cpu]

Inputs come from the environment: COSIR_ARTELINGO_FEATURES and COSIR_ARTELINGO_ANNOTATIONS (read by
src.data.artelingo), CLIPFT_IMAGE_CACHE (ft_data's uint8 image cache; LB and LoRA only), HF_HUB_CACHE / HF_HUB_OFFLINE.

- Sampler: every epoch draws one caption per scorer-train painting, shuffled, seeded by (0, epoch); a batch never holds
  two captions of one painting. Symmetric InfoNCE, learnable temperature initialised from CLIP's logit_scale and
  clamped at log(100); AdamW, linear warm-up over 5% of all steps then cosine decay; bf16 autocast on CUDA for LB
  and LoRA (LP trains in fp32).
- Val retrieval after every epoch (epoch 0 = plain CLIP, the reference): image->caption R@1 over all val captions (a
  hit if the top caption belongs to the image's painting), caption->image R@1 over the val paintings' images; their
  mean is the selection metric. Ties go to the lower index. The best epoch is chosen among epochs >= 1.
- Outputs in --out: metrics.json (rewritten after every epoch), best_params.pt (the best epoch's trained parameters
  only), features.npz (rows int64 = every val and selection row in feature-row order; img, txt float32 (n, 512) =
  raw projection outputs of the best epoch, not normalised), features_epoch0.npz (the same for the untrained model:
  LP = the frozen cached features, LB/LoRA = plain CLIP through the uint8 image cache), run_record.json.
- Held rows are never used: every row the trainer touches is asserted to lie in scorer_train, val or selection; held
  rows' captions and images are never read. No evaluation label is read.
- Rows follow the feature-row order; a row's caption is annotations[data.sample_ids[row]]["caption"]; a row's image
  is the cache row of its painting in load_image_cache's painting index.
"""
import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ft_data  # noqa: E402

CLIP_NAME = ft_data.CLIP_NAME
VARIANTS = ("LP", "LB", "LoRA")
DIM = 512
SEED = 0
BATCH_SIZE = 256
EVAL_BATCH = 256
SIM_CHUNK = 4096
WARMUP_FRACTION = 0.05
MAX_LOGIT_SCALE = math.log(100.0)
LB_WEIGHT_DECAY = 0.1
ADAMW = {"betas": (0.9, 0.999), "eps": 1e-8}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]}
TOKENS = {"max_length": 77, "padding": "max_length", "truncation": True}
SMOKE = {"train_paintings": 64, "val_rows": 32, "selection_rows": 32, "max_epochs": 2, "batch_size": 32}
OUT_FILES = ("metrics.json", "best_params.pt", "features.npz", "features_epoch0.npz", "run_record.json")
_MEAN = torch.tensor(ft_data.CLIP_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(ft_data.CLIP_STD).view(1, 3, 1, 1)


def now() -> str:
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------------------------------------------------ sampler
def epoch_order(painting_pos, epoch: int, seed: int = SEED) -> np.ndarray:
    """One train row per painting, in shuffled order: indices into `painting_pos` (len = number of paintings).

    painting_pos[j] is the painting position (0..P-1, each present) of train row j. The caption of each painting is
    drawn uniformly, then the paintings are shuffled; both use np.random.default_rng((seed, epoch)).
    """
    pos = np.asarray(painting_pos, dtype=np.int64)
    rng = np.random.default_rng((seed, epoch))
    grouped = np.argsort(pos, kind="stable")
    counts = np.bincount(pos)
    if (counts == 0).any():
        raise ValueError("painting positions must cover 0..P-1")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pick = grouped[starts + rng.integers(0, counts)]
    return pick[rng.permutation(len(pick))].astype(np.int64)


def batches(order, batch_size: int) -> list:
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


# ------------------------------------------------------------------------------------------------ rows
def check_rows(rows, idx: dict, what: str = "rows") -> None:
    """Raise ValueError if any row lies outside scorer_train, val and selection (i.e. is held or unknown)."""
    allowed = np.concatenate([np.asarray(idx[k], dtype=np.int64) for k in ft_data.SPLIT_NAMES])
    bad = ~np.isin(np.asarray(rows, dtype=np.int64), allowed)
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} rows outside scorer_train/val/selection (held or unknown)")


def smoke_subset(idx: dict, paintings, available=None, n_paint=SMOKE["train_paintings"], n_val=SMOKE["val_rows"],
                 n_sel=SMOKE["selection_rows"]) -> dict:
    """Smoke rows: all rows of the first n_paint scorer-train paintings (sorted ids), the first n_val val rows and the
    first n_sel selection rows (feature-row order); only paintings in `available` (the image cache) if given."""
    paintings = np.asarray(paintings)

    def usable(rows):
        rows = np.asarray(rows, dtype=np.int64)
        return rows if available is None else rows[np.isin(paintings[rows], np.array(sorted(available)))]

    train = usable(idx["scorer_train"])
    keep = np.unique(paintings[train])[:n_paint]
    return {"scorer_train": train[np.isin(paintings[train], keep)], "val": usable(idx["val"])[:n_val],
            "selection": usable(idx["selection"])[:n_sel]}


@dataclass
class RowSet:
    """Feature rows (sorted) with their paintings and encoder inputs."""
    rows: np.ndarray          # feature rows, ascending
    pos: np.ndarray           # painting position of each row (into `paintings`)
    paintings: np.ndarray     # unique painting ids, sorted
    first: np.ndarray         # index into rows of each painting's first row
    img_feats: np.ndarray = None   # LP: cached frozen features of each row
    txt_feats: np.ndarray = None
    cache_rows: np.ndarray = None  # LB/LoRA: image-cache row of each painting (from the cache's painting index)
    ids: np.ndarray = None         # LB/LoRA: token ids of each row's caption
    mask: np.ndarray = None


def tokenize(tokenizer, texts, chunk: int = 8192):
    """CLIP tokens (max_length 77, padded, truncated) as int32 ids and int8 masks."""
    ids, mask = [np.zeros((0, TOKENS["max_length"]), np.int32)], [np.zeros((0, TOKENS["max_length"]), np.int8)]
    for i in range(0, len(texts), chunk):
        enc = tokenizer(list(texts[i:i + chunk]), return_tensors="np", **TOKENS)
        ids.append(enc["input_ids"].astype(np.int32))
        mask.append(enc["attention_mask"].astype(np.int8))
    return np.concatenate(ids), np.concatenate(mask)


def make_rowset(variant, data, rows, annotations=None, tokenizer=None, cache_index=None) -> RowSet:
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    paintings, pos = ft_data.painting_table(data, rows)
    _, first = np.unique(pos, return_index=True)
    rs = RowSet(rows, pos, paintings, first.astype(np.int64))
    if variant == "LP":
        rs.img_feats = np.ascontiguousarray(data.img_features[rows], dtype=np.float32)
        rs.txt_feats = np.ascontiguousarray(data.txt_features[rows], dtype=np.float32)
        return rs
    missing = [str(p) for p in paintings if str(p) not in cache_index]
    if missing:
        raise ValueError(f"{len(missing)} paintings missing from the image cache, e.g. {missing[:3]}")
    rs.cache_rows = np.array([cache_index[str(p)] for p in paintings], dtype=np.int64)  # painting -> cache row
    rs.ids, rs.mask = tokenize(tokenizer, ft_data.captions(np.asarray(data.sample_ids)[rows], annotations))
    return rs


# ------------------------------------------------------------------------------------------------ models
def load_clip():
    from transformers import CLIPModel
    return CLIPModel.from_pretrained(CLIP_NAME).float()


def load_tokenizer():
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(CLIP_NAME)


class LinearProbe(nn.Module):
    """One 512 x 512 map per modality (no bias), initialised to the identity, on the frozen projection features."""

    def __init__(self, logit_scale: float):
        super().__init__()
        self.img_map = nn.Linear(DIM, DIM, bias=False)
        self.txt_map = nn.Linear(DIM, DIM, bias=False)
        with torch.no_grad():
            self.img_map.weight.copy_(torch.eye(DIM))
            self.txt_map.weight.copy_(torch.eye(DIM))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def encode_image(self, x):
        return self.img_map(x)

    def encode_text(self, x):
        return self.txt_map(x)


def encode_image(model, x):
    """Raw projection output: visual_projection(vision_model(pixel_values).pooler_output) (LP: the map)."""
    if isinstance(model, LinearProbe):
        return model.encode_image(x)
    return model.visual_projection(model.vision_model(pixel_values=x).pooler_output)


def encode_text(model, ids, mask=None):
    if isinstance(model, LinearProbe):
        return model.encode_text(ids)
    return model.text_projection(model.text_model(input_ids=ids, attention_mask=mask).pooler_output)


def lb_prefixes(clip) -> tuple:
    lv = clip.config.vision_config.num_hidden_layers - 1
    lt = clip.config.text_config.num_hidden_layers - 1
    return (f"vision_model.encoder.layers.{lv}.", f"text_model.encoder.layers.{lt}.", "vision_model.post_layernorm.",
            "text_model.final_layer_norm.", "visual_projection.", "text_projection.")


def build_model(variant: str, clip):
    """LP: a new LinearProbe (temperature from clip.logit_scale). LB and LoRA modify `clip` in place and return it.

    LB trains the last encoder layer, post_layernorm / final_layer_norm and the projection of each tower plus
    logit_scale; LoRA trains rank-16 adapters on q/k/v/out_proj of both towers plus logit_scale. The frozen prefix
    of LB carries no autograd graph (no input of it requires grad), so it costs what a no_grad pass costs.
    """
    if variant == "LP":
        return LinearProbe(clip.logit_scale.detach().item())
    clip.requires_grad_(False)
    if variant == "LB":
        prefixes = lb_prefixes(clip)
        for name, p in clip.named_parameters():
            if name == "logit_scale" or name.startswith(prefixes):
                p.requires_grad_(True)
    elif variant == "LoRA":
        from peft import LoraConfig, get_peft_model
        get_peft_model(clip, LoraConfig(**LORA))  # injects the adapters into `clip`; only they require grad
        clip.logit_scale.requires_grad_(True)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return clip


def trainable_names(model) -> set:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_state(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state: dict) -> None:
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise ValueError("saved parameters do not match the variant's trainable set")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n].to(p.device, p.dtype))


def param_groups(model, variant: str) -> list:
    """LB: weight decay 0.1 on weight matrices, none on biases, layer norms and logit_scale. LP, LoRA: none."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if variant != "LB":
        return [{"params": [p for _, p in named], "names": [n for n, _ in named], "weight_decay": 0.0}]
    is_decay = [p.ndim >= 2 and "norm" not in n for n, p in named]
    groups = []
    for wd, sel in ((LB_WEIGHT_DECAY, True), (0.0, False)):
        chosen = [(n, p) for (n, p), d in zip(named, is_decay) if d == sel]
        groups.append({"params": [p for _, p in chosen], "names": [n for n, _ in chosen], "weight_decay": wd})
    return groups


def make_optimizer(model, variant: str, lr: float):
    return torch.optim.AdamW(param_groups(model, variant), lr=lr, **ADAMW)


def schedule(total_steps: int, warmup_fraction: float = WARMUP_FRACTION):
    """(warm-up steps, LambdaLR factor): linear warm-up over ceil(5%) of all steps, then cosine decay to 0."""
    warmup = max(1, math.ceil(warmup_fraction * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return warmup, factor


# ------------------------------------------------------------------------------------------------ training
def to_pixel_values(u8):
    """(N, 224, 224, 3) uint8 -> CLIP-normalised (N, 3, 224, 224) float32 (rescale 1/255, CLIP mean and std)."""
    x = u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def clip_loss(img, txt, logit_scale):
    """Symmetric InfoNCE on L2-normalised features, in fp32."""
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    logits = logit_scale.float().exp() * img @ txt.t()
    target = torch.arange(len(img), device=img.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2


def forward_loss(model, variant, batch, device, amp: bool = False):
    """batch: LP (img_feats, txt_feats); LB/LoRA (uint8 images NHWC, input_ids, attention_mask)."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        if variant == "LP":
            fi = model.encode_image(batch[0].to(device, non_blocking=True))
            ft = model.encode_text(batch[1].to(device, non_blocking=True))
        else:
            u8, ids, mask = batch
            fi = encode_image(model, to_pixel_values(u8.to(device, non_blocking=True)))
            ft = encode_text(model, ids.to(device, non_blocking=True).long(), mask.to(device, non_blocking=True).long())
    return clip_loss(fi, ft, model.logit_scale)


def train_step(model, variant, batch, optimizer, scheduler, device, amp: bool = False) -> float:
    model.train()
    loss = forward_loss(model, variant, batch, device, amp)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    with torch.no_grad():
        model.logit_scale.clamp_(0.0, MAX_LOGIT_SCALE)
    return float(loss.detach())


@torch.no_grad()
def batch_loss(model, variant, batch, device) -> float:
    model.eval()
    return float(forward_loss(model, variant, batch, device))


class TrainPairs(torch.utils.data.Dataset):
    """Item j: (index j, uint8 image of train row j's painting, its caption tokens). The memmap opens lazily."""

    def __init__(self, cache_dir, image_rows, ids, mask):
        self.cache_dir, self.image_rows, self.ids, self.mask = str(cache_dir), image_rows, ids, mask
        self._images = None

    def __getstate__(self):  # never pickle an open memmap into a worker; each worker reopens it
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self):
        return len(self.image_rows)

    def __getitem__(self, j):
        if self._images is None:
            self._images = np.load(Path(self.cache_dir) / ft_data.IMAGES_FILE, mmap_mode="r")
        return (j, torch.from_numpy(np.array(self._images[self.image_rows[j]])),
                torch.from_numpy(self.ids[j].astype(np.int64)), torch.from_numpy(self.mask[j].astype(np.int64)))


def train_dataset(train: RowSet, cache_dir) -> TrainPairs:
    """Training pairs: item j = the image of train.rows[j]'s painting (its cache row from the painting index) and the
    caption tokens of train.rows[j]."""
    return TrainPairs(cache_dir, train.cache_rows[train.pos], train.ids, train.mask)


def use_bf16(variant: str, device) -> bool:
    """bf16 autocast for LB and LoRA on a CUDA device that supports it; LP (a 512 x 512 map) trains in fp32."""
    return variant != "LP" and torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported()


class EpochOrder(torch.utils.data.Sampler):
    """Yields the current epoch's order (set before each pass), so persistent workers see every epoch."""

    def __init__(self):
        self.order = np.zeros(0, dtype=np.int64)

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self):
        return len(self.order)


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def encode_rowset(model, variant, rs: RowSet, device, images=None, batch: int = EVAL_BATCH):
    """Raw projection outputs of every row of rs (fp32, eval mode): (img, txt) float32 CPU tensors (len(rows), 512).

    LB/LoRA encode each painting's image once (cache row from the painting index) and give it to all its rows.
    """
    model.eval()
    if variant == "LP":
        out = []
        for feats, enc in ((rs.img_feats, model.encode_image), (rs.txt_feats, model.encode_text)):
            x = torch.from_numpy(feats)
            out.append(torch.cat([enc(x[i:i + batch].to(device)).float().cpu() for i in range(0, len(x), batch)]
                                 or [torch.zeros(0, DIM)]))
        return out[0], out[1]
    img_p = [torch.zeros(0, DIM)]
    for i in range(0, len(rs.cache_rows), batch):
        u8 = torch.from_numpy(np.ascontiguousarray(images[rs.cache_rows[i:i + batch]]))
        img_p.append(encode_image(model, to_pixel_values(u8.to(device))).float().cpu())
    txt = [torch.zeros(0, DIM)]
    for i in range(0, len(rs.rows), batch):
        ids = torch.from_numpy(rs.ids[i:i + batch].astype(np.int64)).to(device)
        mask = torch.from_numpy(rs.mask[i:i + batch].astype(np.int64)).to(device)
        txt.append(encode_text(model, ids, mask).float().cpu())
    return torch.cat(img_p)[torch.from_numpy(rs.pos)], torch.cat(txt)


def _r1_core(blocks, cap_img, n_img: int) -> dict:
    """blocks: (c0, sim[:, c0:c0+c]) with one row per image. Ties go to the lower caption / image index."""
    cap = torch.as_tensor(np.asarray(cap_img), dtype=torch.long)
    if n_img == 0 or len(cap) == 0:
        raise ValueError("retrieval needs at least one image and one caption")
    best_val = torch.full((n_img,), -math.inf, dtype=torch.float64)
    best_idx = torch.zeros(n_img, dtype=torch.long)
    t2i = 0
    for c0, block in blocks:
        top_img = block.argmax(dim=0).cpu()  # first maximum over images
        t2i += int((top_img == cap[c0:c0 + block.shape[1]]).sum())
        val, idx = block.max(dim=1)          # first maximum within the block
        val, idx = val.double().cpu(), idx.cpu()
        upd = val > best_val                 # strict: an earlier block keeps a tie
        best_val[upd], best_idx[upd] = val[upd], idx[upd] + c0
    i2t = int((cap[best_idx] == torch.arange(n_img)).sum())
    i2t_r1, t2i_r1 = i2t / n_img, t2i / len(cap)
    return {"i2t_r1": i2t_r1, "t2i_r1": t2i_r1, "selection": (i2t_r1 + t2i_r1) / 2, "i2t_correct": i2t,
            "t2i_correct": t2i, "n_images": n_img, "n_captions": len(cap)}


def r1_from_sim(sim, cap_img, chunk: int = SIM_CHUNK) -> dict:
    """R@1 both ways from a similarity matrix (images x captions); cap_img[j] = the image of caption j."""
    sim = torch.as_tensor(sim)
    return _r1_core(((c0, sim[:, c0:c0 + chunk]) for c0 in range(0, sim.shape[1], chunk)), cap_img, sim.shape[0])


def r1_from_features(img, txt, cap_img, device="cpu", chunk: int = SIM_CHUNK) -> dict:
    """Cosine R@1 both ways: img (P, d) one per image, txt (N, d), cap_img (N,) image position of each caption."""
    img = F.normalize(torch.as_tensor(img).float(), dim=-1).to(device)
    txt = F.normalize(torch.as_tensor(txt).float(), dim=-1).to(device)
    return _r1_core(((c0, img @ txt[c0:c0 + chunk].t()) for c0 in range(0, len(txt), chunk)), cap_img, len(img))


def val_retrieval(img_rows, txt_rows, rs: RowSet, device="cpu") -> dict:
    """Val retrieval: one image per painting (its first row's), all captions."""
    return r1_from_features(img_rows[torch.from_numpy(rs.first)], txt_rows, rs.pos, device)


def best_epoch(epochs: list):
    """The epoch (>= 1) with the highest selection metric, ties to the earlier one; epoch 0 is the reference only."""
    cands = [e for e in epochs if e["epoch"] >= 1]
    return min(cands, key=lambda e: (-e["selection"], e["epoch"]))["epoch"] if cands else None


def select_runs(runs: list) -> dict:
    """Spec §4 across one variant's runs (metrics.json dicts with "lr" and "epochs"): the (lr, epoch >= 1) with the
    highest selection metric; ties go to the smaller learning rate, then the earlier epoch."""
    s, lr, ep = min((-e["selection"], float(r["lr"]), e["epoch"]) for r in runs for e in r["epochs"] if e["epoch"] >= 1)
    return {"lr": lr, "epoch": ep, "selection": -s}


def feature_check(img_rows, txt_rows, img_ref, txt_ref) -> dict:
    """Recomputed features vs the cached frozen ones (epoch 0): max abs difference and min cosine per modality."""
    out = {}
    for name, a, b in (("img", img_rows, img_ref), ("txt", txt_rows, txt_ref)):
        b = torch.from_numpy(np.asarray(b, dtype=np.float32))
        out[f"{name}_max_abs_diff"] = float((a - b).abs().max()) if len(a) else None
        out[f"{name}_min_cos"] = float(F.cosine_similarity(a, b, dim=-1).min()) if len(a) else None
    return out


# ------------------------------------------------------------------------------------------------ run
def write_features(path: Path, rowsets, feats, idx: dict) -> np.ndarray:
    """Write the features.npz layout: rows int64 ascending (feature-row order); img, txt float32 (n, 512) raw
    projection outputs. rowsets and feats ((img, txt) per row set) are aligned; returns the rows."""
    rows = np.concatenate([rs.rows for rs in rowsets])
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    if len(np.unique(rows)) != len(rows):
        raise AssertionError(f"{path.name}: duplicate rows")
    check_rows(rows, idx, f"{path.name} rows")
    img = torch.cat([f[0] for f in feats]).numpy()[order].astype(np.float32)
    txt = torch.cat([f[1] for f in feats]).numpy()[order].astype(np.float32)
    if not (np.isfinite(img).all() and np.isfinite(txt).all()):
        raise AssertionError(f"{path.name}: non-finite features")
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, rows=rows.astype(np.int64), img=img, txt=txt)
    tmp.replace(path)
    return rows


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _git_info() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "HEAD") or "unknown",
            "dirty_in_code_folder": git("status", "--porcelain", "--", str(HERE)).splitlines()}


def _versions() -> dict:
    import transformers
    try:
        import peft
        peft_v = peft.__version__
    except ImportError:
        peft_v = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_v,
            "numpy": np.__version__, "python": platform.python_version(), "cuda": torch.version.cuda}


def run(args, data=None, annotations=None, splits=None, cache_dir=None, clip=None, verbose: bool = True) -> dict:
    """Train one (variant, lr) and write the five output files. data/annotations/splits/cache_dir/clip default to the
    real inputs (env paths); tests pass toy ones."""
    def log(*a):
        if verbose:
            print(*a, flush=True)

    t0, start = time.time(), now()
    out = Path(args.out)
    if any((out / f).exists() for f in OUT_FILES):
        raise FileExistsError(f"{out} already holds run outputs; clear --out first (remove the folder) to rerun")
    out.mkdir(parents=True, exist_ok=True)
    variant, device = args.variant, torch.device(args.device)
    image_variant = variant != "LP"
    epochs = min(args.epochs, SMOKE["max_epochs"]) if args.smoke else args.epochs
    batch_size = SMOKE["batch_size"] if args.smoke else BATCH_SIZE
    amp = use_bf16(variant, device)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from src.data import artelingo as art
    if data is None:
        log(f"[ft] loading features from {art.FEATURE_DIR}")
        data = art.load_artelingo()
    if annotations is None and image_variant:
        annotations = json.loads(Path(art.ANNOTATIONS_PATH).read_text())
    idx = ft_data.split_index(data, splits=splits)
    images = cache_index = cache_record = None
    if image_variant:
        cache_dir = Path(cache_dir or os.environ.get("CLIPFT_IMAGE_CACHE") or ft_data.DEFAULT_CACHE_DIR)
        log(f"[ft] image cache {cache_dir} (verifying SHA-256)")
        images, cache_index, cache_record = ft_data.load_image_cache(cache_dir, verify=True)
    use = smoke_subset(idx, data.paintings, set(cache_index) if image_variant else None) if args.smoke else idx
    for k in ft_data.SPLIT_NAMES:
        check_rows(use[k], idx, k)

    clip = load_clip() if clip is None else clip
    tokenizer = load_tokenizer() if image_variant else None
    sets = {k: make_rowset(variant, data, use[k], annotations, tokenizer, cache_index) for k in ft_data.SPLIT_NAMES}
    train, val, sel = sets["scorer_train"], sets["val"], sets["selection"]
    cached_val = (data.img_features[val.rows], data.txt_features[val.rows])
    annotations = None  # captions are tokenised; nothing else of the annotations is needed
    model = build_model(variant, clip).to(device)
    names = sorted(trainable_names(model))
    log(f"[ft] {variant} lr {args.lr}: {n_trainable(model):,} trainable parameters in {len(names)} tensors; "
        f"train {len(train.rows)} rows / {len(train.paintings)} paintings, val {len(val.rows)} rows / "
        f"{len(val.paintings)} paintings, selection {len(sel.rows)} rows; device {device}, bf16 {amp}")

    steps_per_epoch = math.ceil(len(train.paintings) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup, factor = schedule(total_steps)
    optimizer = make_optimizer(model, variant, args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    sampler, loader, lp_train = EpochOrder(), None, None
    if image_variant:
        workers = min(int(args.workers), 8)
        loader = torch.utils.data.DataLoader(
            train_dataset(train, cache_dir), batch_size=batch_size,
            sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda", drop_last=False,
            persistent_workers=workers > 0, multiprocessing_context="spawn" if workers > 0 else None)
    else:
        lp_train = (torch.from_numpy(train.img_feats).to(device), torch.from_numpy(train.txt_feats).to(device))

    metrics = {"variant": variant, "lr": args.lr, "smoke": bool(args.smoke),
               "selection_metric": "mean of val image->caption R@1 and caption->image R@1; ties to the lower index",
               "n_val_rows": int(len(val.rows)), "n_val_paintings": int(len(val.paintings)),
               "epochs": [], "best_epoch": None, "best_selection": None}
    checks, best_state = {}, None

    def evaluate(epoch, train_loss, seconds):
        img_rows, txt_rows = encode_rowset(model, variant, val, device, images)
        r = val_retrieval(img_rows, txt_rows, val, device)
        rec = {"epoch": epoch, **r, "train_loss": train_loss, "logit_scale": float(model.logit_scale.detach()),
               "lr_end": optimizer.param_groups[0]["lr"] if epoch else None, "seconds": seconds, "time": now()}
        metrics["epochs"].append(rec)
        return rec, img_rows, txt_rows

    rec, img0, txt0 = evaluate(0, None, 0.0)
    checks["epoch0_vs_cached_val_features"] = feature_check(img0, txt0, *cached_val)
    log(f"[ft] epoch 0 (plain CLIP) evaluated; recomputed vs cached val features: "
        f"{checks['epoch0_vs_cached_val_features']}")
    write_features(out / "features_epoch0.npz", (val, sel),
                   ((img0, txt0), encode_rowset(model, variant, sel, device, images)), idx)
    _write_json(out / "metrics.json", metrics)

    step = 0
    for epoch in range(1, epochs + 1):
        te = time.time()
        order = epoch_order(train.pos, epoch)
        chunks = batches(order, batch_size)
        sampler.order = order
        stream = iter(loader) if loader is not None else (None for _ in chunks)
        losses = []
        for b, item in zip(chunks, stream):
            if len(np.unique(train.pos[b])) != len(b):
                raise AssertionError("a batch holds two captions of one painting")
            if variant == "LP":
                bt = torch.from_numpy(b).to(device)
                batch = (lp_train[0][bt], lp_train[1][bt])
            else:
                js, u8, ids, mask = item
                if not np.array_equal(js.numpy(), b):
                    raise AssertionError("the loader's batch differs from the sampler's")
                batch = (u8, ids, mask)
            losses.append(train_step(model, variant, batch, optimizer, scheduler, device, amp))
            step += 1
            if verbose and not args.smoke and step % 50 == 0:
                log(f"[ft] step {step}/{total_steps} loss {losses[-1]:.4f} lr {optimizer.param_groups[0]['lr']:.3g}")
        if len(losses) != len(chunks):
            raise AssertionError("the loader ended early")
        rec, _, _ = evaluate(epoch, float(np.mean(losses)), round(time.time() - te, 1))
        if metrics["best_selection"] is None or rec["selection"] > metrics["best_selection"]:
            metrics["best_epoch"], metrics["best_selection"] = epoch, rec["selection"]
            best_state = trainable_state(model)
            tmp = out / "best_params.pt.part"
            torch.save({"variant": variant, "lr": args.lr, "epoch": epoch, "params": best_state}, tmp)
            tmp.replace(out / "best_params.pt")
        _write_json(out / "metrics.json", metrics)
        if args.smoke:
            log(f"[ft] epoch {epoch}/{epochs} done ({rec['seconds']}s); metrics in metrics.json")
        else:
            log(f"[ft] epoch {epoch}/{epochs}: loss {rec['train_loss']:.4f} i2t {rec['i2t_r1']:.4f} "
                f"t2i {rec['t2i_r1']:.4f} selection {rec['selection']:.4f} ({rec['seconds']}s); "
                f"best epoch {metrics['best_epoch']}")
    if best_epoch(metrics["epochs"]) != metrics["best_epoch"]:
        raise AssertionError("best epoch bookkeeping disagrees with the tie rule")

    # Features of the best epoch for every val and selection row, in feature-row order.
    load_trainable(model, best_state)
    vi, vt = encode_rowset(model, variant, val, device, images)
    si, st = encode_rowset(model, variant, sel, device, images)
    again = val_retrieval(vi, vt, val, device)
    best_rec = metrics["epochs"][metrics["best_epoch"]]
    diff = {k: again[k] - best_rec[k] for k in ("i2t_correct", "t2i_correct")}
    checks["reload"] = {"ok": all(abs(v) <= 2 for v in diff.values()), "count_differences": diff,
                        "note": "val retrieval recomputed from best_params equals the best epoch's (<= 2 near-tie flips)"}
    rows = write_features(out / "features.npz", (val, sel), ((vi, vt), (si, st)), idx)

    git = _git_info()
    record = {
        "args": vars(args), "variant": variant, "lr": args.lr, "smoke": bool(args.smoke),
        "git_commit": git["commit"], "git_dirty_in_code_folder": git["dirty_in_code_folder"],
        "hostname": socket.gethostname(), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "versions": _versions(), "start_time": start, "end_time": now(), "duration_s": round(time.time() - t0, 1),
        "best_epoch": metrics["best_epoch"], "best_selection": metrics["best_selection"],
        "n_trainable_params": n_trainable(model), "trainable_tensors": len(names),
        "data": {"features_dir": str(art.FEATURE_DIR), "annotations_path": str(art.ANNOTATIONS_PATH),
                 "image_cache_dir": str(cache_dir) if image_variant else None,
                 "image_cache_sha256": cache_record["sha256"] if cache_record else None,
                 "n_train_rows": int(len(train.rows)), "n_train_paintings": int(len(train.paintings)),
                 "n_val_rows": int(len(val.rows)), "n_val_paintings": int(len(val.paintings)),
                 "n_selection_rows": int(len(sel.rows)), "features_rows": int(len(rows))},
        "training": {"epochs": epochs, "batch_size": batch_size, "steps_per_epoch": steps_per_epoch,
                     "total_steps": total_steps, "warmup_steps": warmup, "schedule": "linear warm-up, cosine to 0",
                     "optimizer": "AdamW", **{k: list(v) if isinstance(v, tuple) else v for k, v in ADAMW.items()},
                     "weight_decay": {g_["weight_decay"]: len(g_["names"]) for g_ in optimizer.param_groups},
                     "amp": "bf16" if amp else "fp32", "eval_precision": "fp32", "seed": SEED,
                     "sampler": "one caption per painting per epoch, default_rng((0, epoch))",
                     "logit_scale_max": MAX_LOGIT_SCALE, "lora": LORA if variant == "LoRA" else None,
                     "tokens": TOKENS, "workers": int(args.workers) if image_variant else 0},
        "checks": checks,
    }
    _write_json(out / "run_record.json", record)
    log(f"[ft] done: best epoch {metrics['best_epoch']}; wrote {', '.join(OUT_FILES)} to {out}")
    if not checks["reload"]["ok"]:
        raise AssertionError(f"reloaded best parameters do not reproduce the best epoch: {diff}")
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Lightweight CLIP fine-tuning on ArtELingo (LP, LB, LoRA).")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--lr", required=True, type=float)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="64 scorer-train paintings, 32 val and 32 selection rows, at most 2 epochs, batch 32")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers for LB/LoRA (at most 8)")
    a = ap.parse_args(argv)
    if a.epochs < 1:
        ap.error("--epochs must be >= 1")
    if not 0 <= a.workers <= 8:
        ap.error("--workers must be in 0..8")
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda but no GPU is visible")
    return a


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
