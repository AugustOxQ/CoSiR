import torch
import torch.nn as nn

# Load pretrained model
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import Optional

from src.model.combiner import *

_DEFAULT_BACKBONE = "openai/clip-vit-base-patch32"


def get_backbone(model_name: str = _DEFAULT_BACKBONE, trainable: bool = False):
    """Load any CLIP-like backbone (CLIP, SigLIP, SigLIP2, etc.) from Hugging Face."""
    backbone = AutoModel.from_pretrained(model_name)
    if not trainable:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone


def get_backbone_feature_dim(backbone) -> int:
    """Auto-detect the joint embedding dimension of a CLIP-like backbone.

    Detection order:
    1. config.projection_dim  — CLIP family (openai/clip-*, laion/CLIP-*)
    2. visual_projection.out_features — other models with an explicit top-level proj
    3. vision_model.config.hidden_size — SigLIP / SigLIP2 (projection is inside the
       vision encoder head, pooler_output already has the final dim)
    """
    # 1. CLIP-style: projection_dim in the top-level config
    if hasattr(backbone, "config") and hasattr(backbone.config, "projection_dim"):
        return backbone.config.projection_dim
    # 2. Explicit top-level projection layer
    if hasattr(backbone, "visual_projection"):
        proj = backbone.visual_projection
        if isinstance(proj, nn.Linear):
            return proj.out_features
    # 3. SigLIP / SigLIP2: the vision encoder head outputs hidden_size directly
    if hasattr(backbone, "vision_model") and hasattr(backbone.vision_model, "config"):
        hidden = getattr(backbone.vision_model.config, "hidden_size", None)
        if hidden is not None:
            return hidden
    raise ValueError(
        f"Cannot auto-detect feature dim for backbone of type {type(backbone).__name__}. "
        "Please set it explicitly or use a model with config.projection_dim."
    )


class CoSiRModel(nn.Module):
    """CoSiR model supporting any CLIP-like backbone (CLIP, SigLIP, SigLIP2, …)."""

    def __init__(
        self,
        backbone_model: str = _DEFAULT_BACKBONE,
        backbone_trainable: bool = False,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        label_dim: int = 32,
        num_conditions: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Load backbone and detect its feature dimension
        self.backbone = get_backbone(backbone_model, backbone_trainable)
        self.feature_dim = get_backbone_feature_dim(self.backbone)

        # Vision encoder (+ optional separate projection layer)
        # CLIP: vision_model outputs raw CLS token, visual_projection maps it to joint space
        # SigLIP: vision_model.head already projects, pooler_output IS the final embedding
        self.vision_model = self.backbone.vision_model  # type: ignore
        self.visual_projection = getattr(self.backbone, "visual_projection", None)

        # Text encoder (+ optional separate projection layer)
        self.text_model = self.backbone.text_model  # type: ignore
        self.text_projection = getattr(self.backbone, "text_projection", None)

        # Identity function for now
        self.label_encoder = nn.Identity()

        # Combiner network to combine text and label features
        self.combiner = Combiner_new(
            clip_feature_dim=self.feature_dim,
            projection_dim=self.feature_dim,
            label_dim=label_dim,
            hidden_dim=d_model,
            num_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Additional components
        # self.unified_condition_predictor = ConditionClassifier(
        #     input_dim=512,
        #     hidden_dim=512,  # Here the dim is fixed to 512, because the input dim is 512
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     num_conditions=num_conditions,
        #     use_temperature=True,
        #     init_temperature=1.0,
        # )

        self.pretrained_representatives = None

    def set_pretrained_representatives(self, representatives: Tensor, device: str):
        self.pretrained_representatives = representatives.to(device)

    def encode_img(self, images):
        img_output = self.vision_model(**images)
        if self.visual_projection is not None:
            # CLIP-style: pooler_output is raw CLS token, needs projection
            img_emb = self.visual_projection(img_output.pooler_output)
            img_full = self.visual_projection(img_output.last_hidden_state)
        else:
            # SigLIP-style: pooler_output is already the projected embedding
            img_emb = img_output.pooler_output
            img_full = img_output.last_hidden_state
        return img_emb, img_full

    def encode_txt(self, texts):
        txt_output = self.text_model(**texts)
        if self.text_projection is not None:
            # CLIP-style
            txt_emb = self.text_projection(txt_output.pooler_output)
            txt_full = self.text_projection(txt_output.last_hidden_state)
        else:
            # SigLIP-style
            txt_emb = txt_output.pooler_output
            txt_full = txt_output.last_hidden_state
        return txt_emb, txt_full

    def encode_img_txt(self, images, texts):
        # Extract image and text features

        img_emb, img_full = self.encode_img(images)
        txt_emb, txt_full = self.encode_txt(texts)

        return img_emb, txt_emb, img_full, txt_full

    def combine(self, txt_emb, txt_full, labels, epoch=None, return_label_proj=False, return_delta=False):
        lbl_emb = self.label_encoder(labels)  # (batch_size, 512)
        result = self.combiner(
            txt_emb,
            None,
            lbl_emb,
            return_delta=return_delta,
        )
        return result

    # def predict_condition(
    #     self,
    #     img_emb: Optional[Tensor],
    #     txt_emb: Optional[Tensor],
    #     type: str,
    #     return_logits: bool = True,
    #     training_phase: bool = False,
    #     argmax: bool = False,
    # ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    #     # input_features: (batch_size, d_model), pre_trained_representatives: (num_conditions, 2)
    #     if type == "img":
    #         probs, logits = self.unified_condition_predictor(
    #             img_emb=None,
    #             txt_emb=txt_emb,
    #             return_logits=True,
    #             training_phase=training_phase,
    #             argmax=argmax,
    #         )
    #     elif type == "txt":
    #         probs, logits = self.unified_condition_predictor(
    #             img_emb=img_emb,
    #             txt_emb=None,
    #             return_logits=True,
    #             training_phase=training_phase,
    #             argmax=argmax,
    #         )
    #     elif type == "imgtxt":
    #         probs, logits = self.unified_condition_predictor(
    #             img_emb=img_emb,
    #             txt_emb=txt_emb,
    #             return_logits=True,
    #             training_phase=training_phase,
    #             argmax=argmax,
    #         )
    #     else:
    #         raise ValueError(f"Invalid condition type: {type}")

    #     output = (
    #         probs @ self.pretrained_representatives
    #     )  # [B, num_conditions] @ [num_conditions, 2] -> [B, 2]

    #     if return_logits:
    #         return output, logits
    #     else:
    #         return output

    def forward(self, images, texts, labels):
        # Extract image and text features

        img_emb, txt_emb, _, txt_full = self.encode_img_txt(images, texts)

        # Encode the labels
        lbl_emb = self.label_encoder(labels)  # (batch_size, 512)

        # Combine text and label features
        comb_emb = self.combiner(txt_emb, txt_full, lbl_emb)  # (batch_size, 512)

        return (
            img_emb,
            txt_emb,
            txt_full,
            lbl_emb,
            comb_emb,
        )  # For now we only need img_emb and comb_emb to calculate the loss


def main():
    backbone_name = _DEFAULT_BACKBONE
    processor = AutoProcessor.from_pretrained(backbone_name)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

    model = CoSiRModel(backbone_model=backbone_name)
    label = torch.randn(2, model.feature_dim)

    image_input = processor(images=[image, image], return_tensors="pt")
    text_input = processor(
        ["a photo of a cat", "a photo of a dog"], return_tensors="pt"
    )

    img_emb, txt_emb, txt_full, lbl_emb, comb_emb = model(image_input, text_input, label)
    print(img_emb.shape, txt_emb.shape, lbl_emb.shape, comb_emb.shape)


if __name__ == "__main__":
    main()
