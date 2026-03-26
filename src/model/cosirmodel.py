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
        dropout: float = 0.1,
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

        # Identity function for now
        self.label_encoder = nn.Identity()

        # Combiner network to combine text and label features
        self.combiner = Combiner_new(
            clip_feature_dim=512,
            projection_dim=512,
            hidden_dim=d_model,
            num_heads=nhead,
            num_layers=4,  # TODO: here we fix it to 2
        )

        # Additional components
        self.unified_condition_predictor = ConditionClassifier(
            input_dim=512,
            hidden_dim=512,  # Here the dim is fixed to 512, because the input dim is 512
            num_layers=num_layers,
            dropout=dropout,
            num_conditions=num_conditions,
            use_temperature=True,
            init_temperature=1.0,
        )

        self.pretrained_representatives = None

    def set_pretrained_representatives(self, representatives: Tensor, device: str):
        self.pretrained_representatives = representatives.to(device)

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

    def combine(self, txt_emb, txt_full, labels, epoch=None, return_label_proj=False):
        # Encode the labels
        lbl_emb = self.label_encoder(labels)  # (batch_size, 512)
        comb_emb = self.combiner(
            txt_emb,
            None,
            lbl_emb,
        )  # (batch_size, 512)

        return comb_emb

    def predict_condition(
        self,
        img_emb: Optional[Tensor],
        txt_emb: Optional[Tensor],
        type: str,
        return_logits: bool = True,
        training_phase: bool = False,
        argmax: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # input_features: (batch_size, d_model), pre_trained_representatives: (num_conditions, 2)
        if type == "img":
            probs, logits = self.unified_condition_predictor(
                img_emb=None,
                txt_emb=txt_emb,
                return_logits=True,
                training_phase=training_phase,
                argmax=argmax,
            )
        elif type == "txt":
            probs, logits = self.unified_condition_predictor(
                img_emb=img_emb,
                txt_emb=None,
                return_logits=True,
                training_phase=training_phase,
                argmax=argmax,
            )
        elif type == "imgtxt":
            probs, logits = self.unified_condition_predictor(
                img_emb=img_emb,
                txt_emb=txt_emb,
                return_logits=True,
                training_phase=training_phase,
                argmax=argmax,
            )
        else:
            raise ValueError(f"Invalid condition type: {type}")

        output = (
            probs @ self.pretrained_representatives
        )  # [B, num_conditions] @ [num_conditions, 2] -> [B, 2]

        if return_logits:
            return output, logits
        else:
            return output

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
