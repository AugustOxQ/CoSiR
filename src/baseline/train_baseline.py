import argparse
import math
import os
import sys
import json
import random

# Ensure project root is on sys.path when script is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import wandb

from src.eval.metrics import RecallMetrics
from src.eval.config import EvaluationConfig
from src.dataset import CoSiRValidationDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DEFAULTS = {
    "coco": {
        "train_annotation_path": "/data/SSD/coco/annotations/coco_karpathy_train.json",
        "train_image_path":      "/data/SSD/coco/images",
        "test_annotation_path":  "/data/SSD/coco/annotations/coco_karpathy_test.json",
        "test_image_path":       "/data/SSD/coco/images",
    },
    "redcaps": {
        "train_annotation_path": "/data/PDD/redcaps/redcaps_plus/redcaps_medium.json",
        "train_image_path":      "/data/PDD",
        "test_annotation_path":  "/data/PDD/redcaps/redcaps_plus/redcaps_test.json",
        "test_image_path":       "/data/PDD",
    },
    "impressions": {
        "train_annotation_path": "/project/Impressions/metadata/impressions_train.json",
        "train_image_path":      "/project/Impressions/media",
        "test_annotation_path":  "/project/Impressions/metadata/impressions_test.json",
        "test_image_path":       "/project/Impressions/media",
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Model ─────────────────────────────────────────────────────────────────────
class CLIPBaseline(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        mode: str = "linear",
        num_blocks: int = 2,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.backbone = AutoModel.from_pretrained(clip_model_name)

        # CLIP exposes top-level projection layers; SigLIP bakes them into the
        # encoder head (vision_model.head / text_model.head).  Store None when
        # absent so encode_image/encode_text can branch without attribute errors.
        self._visual_proj = getattr(self.backbone, "visual_projection", None)
        self._text_proj = getattr(self.backbone, "text_projection", None)

        if mode == "full":
            pass  # all params trainable by default

        elif mode == "last_blocks":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for layer in self.backbone.vision_model.encoder.layers[-num_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for layer in self.backbone.text_model.encoder.layers[-num_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self._visual_head_params():
                p.requires_grad = True
            for p in self._text_head_params():
                p.requires_grad = True
            vln = getattr(self.backbone.vision_model, "post_layernorm", None)
            if vln is not None:
                for p in vln.parameters():
                    p.requires_grad = True
            tln = getattr(self.backbone.text_model, "final_layer_norm", None)
            if tln is not None:
                for p in tln.parameters():
                    p.requires_grad = True

        elif mode == "linear":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self._visual_head_params():
                p.requires_grad = True
            for p in self._text_head_params():
                p.requires_grad = True

        elif mode == "lora":
            from peft import get_peft_model, LoraConfig
            for p in self.backbone.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            # Re-resolve after wrapping because get_peft_model replaces the module
            self._visual_proj = getattr(self.backbone, "visual_projection", None)
            self._text_proj = getattr(self.backbone, "text_projection", None)
            for p in self._visual_head_params():
                p.requires_grad = True
            for p in self._text_head_params():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from full|last_blocks|linear|lora")

    def _visual_head_params(self):
        """Projection parameters for the vision side (CLIP or SigLIP)."""
        if self._visual_proj is not None:
            return self._visual_proj.parameters()
        head = getattr(self.backbone.vision_model, "head", None)
        return head.parameters() if head is not None else iter([])

    def _text_head_params(self):
        """Projection parameters for the text side (CLIP or SigLIP)."""
        if self._text_proj is not None:
            return self._text_proj.parameters()
        head = getattr(self.backbone.text_model, "head", None)
        return head.parameters() if head is not None else iter([])

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone.vision_model(pixel_values=pixel_values)
        # SigLIP2 NaViT has no CLS token → pooler_output is None; fall back to mean pool
        feat = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        if self._visual_proj is not None:
            feat = self._visual_proj(feat)
        return feat

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.backbone.text_model(input_ids=input_ids, attention_mask=attention_mask)
        if out.pooler_output is not None:
            feat = out.pooler_output
        elif attention_mask is not None:
            # mask-aware mean pool to ignore padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            feat = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            feat = out.last_hidden_state.mean(dim=1)
        if self._text_proj is not None:
            feat = self._text_proj(feat)
        return feat

    def forward(self, inputs: dict) -> tuple:
        img_emb = self.encode_image(inputs["pixel_values"])
        txt_emb = self.encode_text(inputs["input_ids"], inputs.get("attention_mask"))
        return img_emb, txt_emb

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Dataset ───────────────────────────────────────────────────────────────────
class AnnotationDataset(Dataset):
    def __init__(self, annotation_path: str, image_path: str, ratio: float = 1.0):
        with open(annotation_path) as f:
            annotations = json.load(f)
        n = max(1, int(len(annotations) * ratio))
        self.annotations = annotations[:n]
        self.image_path = image_path

        cap0 = self.annotations[0]["caption"]
        self.captions_per_image = 1 if isinstance(cap0, str) else len(cap0)

        if self.captions_per_image > 1:
            print(
                f"[AnnotationDataset] Multi-caption dataset detected ({self.captions_per_image} captions/image). "
                f"Training uses only the first caption per image."
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.image_path, ann["image"])).convert("RGB")
        cap = ann["caption"] if isinstance(ann["caption"], str) else ann["caption"][0]
        return img, cap


# ── Loss ──────────────────────────────────────────────────────────────────────
def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    logits = img @ txt.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_eval(model, processor, val_loader, device, k_vals=(1, 5, 10)):
    """Compute I2T/T2I recall@K, mAP, meanR for the validation set."""
    model.eval()
    all_img, all_txt = [], []
    t2i_map_list, i2t_map_list = [], []
    text_idx = img_idx = 0
    cpi = getattr(val_loader.dataset, "captions_per_image", 1)

    with torch.no_grad():
        for images, raw_texts in tqdm(val_loader, desc="Eval"):
            B = images["pixel_values"].shape[0]

            # CoSiRValidationDataset collation: when cpi>1, raw_texts is a list of
            # cpi lists each of length B (default_collate transposes the list-of-lists)
            if cpi == 1:
                flat_texts = list(raw_texts)
            else:
                flat_texts = [raw_texts[i][b] for b in range(B) for i in range(cpi)]

            text_inputs = processor(
                text=flat_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(device)

            img_emb = model.encode_image(images["pixel_values"].to(device))
            txt_emb = model.encode_text(
                text_inputs["input_ids"], text_inputs.get("attention_mask")
            )

            all_img.append(img_emb.cpu())
            all_txt.append(txt_emb.cpu())

            for _ in range(B):
                i2t_map_list.append(list(range(text_idx, text_idx + cpi)))
                t2i_map_list.extend([img_idx] * cpi)
                text_idx += cpi
                img_idx += 1

            del img_emb, txt_emb
            torch.cuda.empty_cache()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    t2i_map = torch.LongTensor(t2i_map_list).to(device)
    i2t_map = torch.LongTensor(i2t_map_list).to(device)

    eval_cfg = EvaluationConfig(device=device, k_vals=list(k_vals), print_metrics=True)
    return RecallMetrics(eval_cfg).compute_all_recalls(img_emb, txt_emb, t2i_map, i2t_map)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.clip_model, use_fast=False)

    model = CLIPBaseline(
        args.clip_model, args.mode, args.num_blocks, args.lora_rank, args.lora_alpha
    ).to(device)
    print(f"[{args.mode}] Trainable params: {model.trainable_param_count():,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = AnnotationDataset(
        args.train_annotation_path, args.train_image_path, ratio=args.ratio
    )

    def collate_fn(batch):
        images, texts = zip(*batch)
        return processor(
            images=list(images),
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    val_dataset = CoSiRValidationDataset(
        args.test_annotation_path, args.test_image_path, processor, ratio=1.0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.05,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_name = args.wandb_name or f"{args.mode}_{args.dataset}_seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        config=vars(args),
        mode=args.wandb_mode,
    )

    # ── Train Loop ────────────────────────────────────────────────────────────
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            img_emb, txt_emb = model(batch)
            loss = clip_loss(img_emb, txt_emb, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
                "step": step,
            })
            epoch_loss += loss.item()
            step += 1

        print(f"Epoch {epoch + 1}: avg loss = {epoch_loss / len(train_loader):.4f}")

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = run_eval(model, processor, val_loader, device, args.k_vals)
            wandb.log({"epoch": epoch + 1, **{f"eval/{k}": v for k, v in metrics.items()}})

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    run_name_safe = run_name.replace("/", "_")
    if args.mode == "lora":
        lora_dir = os.path.join(args.output_dir, run_name_safe + "_lora")
        model.backbone.save_pretrained(lora_dir)
        print(f"Saved LoRA adapters to {lora_dir}")
    else:
        ckpt_path = os.path.join(args.output_dir, f"{run_name_safe}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="CLIP baseline fine-tuning")

    # Required
    p.add_argument("--dataset", required=True, choices=["coco", "redcaps", "impressions"])
    p.add_argument("--mode", required=True, choices=["full", "last_blocks", "linear", "lora"])
    p.add_argument("--train_annotation_path", default=None,
                   help="Defaults to DATASET_DEFAULTS[dataset] if omitted")
    p.add_argument("--train_image_path", default=None)
    p.add_argument("--test_annotation_path", default=None)
    p.add_argument("--test_image_path", default=None)

    # Model
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ratio", type=float, default=1.0, help="Fraction of training data to use")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--k_vals", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="res/baseline")
    p.add_argument("--temperature", type=float, default=0.07)

    # WandB
    p.add_argument("--wandb_project", default="cosir_baseline")
    p.add_argument("--wandb_entity", default="augustoxq")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_tags", nargs="*", default=["baseline"])
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"],
                   help="Set to 'disabled' to skip wandb logging entirely")

    args = p.parse_args()

    # Fill in any unspecified paths from the per-dataset defaults
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    return args


if __name__ == "__main__":
    train(parse_args())
