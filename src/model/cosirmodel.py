import torch
import torch.nn as nn
from transformers import CLIPModel

# Load pretrained model
import requests
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from typing import Optional

from src.model.combiner import *


def get_clip(trainable=False):
    """Get CLIP model from Hugging Face's Transformers library."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Set parameters to be non-trainable
    if not trainable:
        for param in model.parameters():  # type: ignore
            param.requires_grad = False
    return model


class CoSiRModel(nn.Module):
    """CLIP-based CoSiR model."""

    def __init__(
        self,
        clip_trainable: bool = False,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        label_dim: int = 32,
        num_conditions: int = 12,
        prototype_conditions: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        # Frozen CLIP as feature extractor
        self.clip = get_clip(clip_trainable)

        # Vision Model
        self.clip_vm = self.clip.vision_model  # type: ignore
        self.clip_vproj = self.clip.visual_projection  # type: ignore

        # Textual Model
        self.clip_tm = self.clip.text_model  # type: ignore
        self.clip_tproj = self.clip.text_projection  # type: ignore

        # Combiner network to combine text and label features
        self.combiner = ConditionEncoder(
            prototype_conditions=prototype_conditions,  # type: ignore
            dim=512,
            K=num_conditions,
            num_heads=nhead,
            dropout=0.1,
            stop_gradient=False,
        )

    def encode_img(self, images):
        # Extract image features
        img_output = self.clip_vm(**images)
        img_emb = self.clip_vproj(img_output.pooler_output)  # CLS token
        img_full = self.clip_vproj(
            img_output.last_hidden_state
        )  # Full image embeddings

        return img_emb, img_full

    def encode_txt(self, texts):
        # Extract text features
        txt_output = self.clip_tm(**texts)
        txt_emb = self.clip_tproj(txt_output.pooler_output)  # CLS token
        txt_full = self.clip_tproj(txt_output.last_hidden_state)  # Full text embeddings

        return txt_emb, txt_full

    def encode_img_txt(self, images, texts):
        # Extract image and text features

        img_emb, img_full = self.encode_img(images)
        txt_emb, txt_full = self.encode_txt(texts)

        return img_emb, txt_emb, img_full, txt_full

    def combine(self, txt_emb, img_emb, img_full):
        # Encode the labels
        comb_emb = self.combiner(
            txt_emb,
            img_emb,
            img_full,
        )  # (batch_size, 512)

        return comb_emb

    def get_condition_from_img(self, img_emb) -> torch.Tensor:

        B = img_emb.size(0)
        Q = self.combiner.conditions.unsqueeze(0).expand(B, -1, -1)  # [B, K, dim]
        KV = img_emb  # [B, 1, dim]

        dynamic_conditions, _ = self.combiner.cross_attn(
            query=Q,
            key=KV,
            value=KV,
        )  # [B, K, dim] # This is a bit problematic

        scores = self.combiner.condition_scorer(dynamic_conditions)  # [B, K, 1]
        attn_weights = F.softmax(scores.squeeze(-1), dim=-1)  # [B, K]
        condition = torch.einsum(
            "bk,bkd->bd", attn_weights, dynamic_conditions
        )  # [B, dim]

        return condition

    def apply_condition_to_txt(self, txt_emb, condition) -> torch.Tensor:

        # 自动广播，不手动expand
        gamma = self.combiner.gamma_net(
            torch.cat([txt_emb, condition.expand_as(txt_emb)], dim=-1)
        )
        beta = self.combiner.beta_net(condition.expand_as(txt_emb))

        return self.combiner.layer_norm(txt_emb + gamma * txt_emb + beta)

    def forward(self, images, texts, labels):
        # Extract image and text features

        img_emb, txt_emb, img_full, txt_full = self.encode_img_txt(images, texts)

        # Combine text and label features
        comb_emb = self.combine(txt_emb, img_emb)  # (batch_size, 512)

        return (
            img_emb,
            txt_emb,
            img_full,
            txt_full,
            comb_emb,
        )  # For now we only need img_emb and comb_emb to calculate the loss


def main():

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)  # type: ignore
    label = torch.randn(2, 512)

    image_input = processor(images=[image, image], return_tensors="pt")
    text_input = processor(
        ["a photo of a cat", "a photo of a dog"], return_tensors="pt"
    )

    model = CoSiRModel()

    img_emb, txt_emb, lbl_emb, comb_emb = model(image_input, text_input, label)
    print(img_emb.shape, txt_emb.shape, lbl_emb.shape, comb_emb.shape)


if __name__ == "__main__":
    main()
