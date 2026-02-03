import torch
import torch.nn as nn
from transformers import CLIPModel

# Load pretrained model
import requests
from PIL import Image
from transformers import AutoProcessor, CLIPModel

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
        self.combiner = CombinerSimplePolar(
            clip_feature_dim=512,
            projection_dim=512,
            hidden_dim=d_model,
            num_heads=nhead,
            num_layers=num_layers,
            label_dim=label_dim,
        )

        # Additional components
        self.img_condition_predictor = ConditionPredictor(
            input_dim=512,
            hidden_dim=256,
            output_dim=2,
            num_layers=num_layers,
        )

        self.txt_condition_predictor = ConditionPredictor(
            input_dim=512,
            hidden_dim=256,
            output_dim=2,
            num_layers=num_layers,
        )

        self.imgtxt_condition_predictor = ConditionPredictor(
            input_dim=512 * 2,
            hidden_dim=256,
            output_dim=2,
            num_layers=num_layers,
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

    def combine(self, txt_emb, txt_full, labels, epoch=None, return_label_proj=False):
        # Encode the labels
        lbl_emb = self.label_encoder(labels)  # (batch_size, 512)
        comb_emb = self.combiner(
            txt_emb,
            None,
            lbl_emb,
            epoch,
            return_label_proj,  # Here txt_full is not used
        )  # (batch_size, 512)

        return comb_emb

    def predict_condition(self, input_features: Tensor, type: str) -> Tensor:
        # BUG: 这里我设计的是如果我们要做text-to-image retrieval，那么我们根据看到的text选择最适合的image。 但是实际还有另一种可能，在做text-to-image retrieval的时候，我们也可以直接反其道而行之，我们根据看到的image选择最适合的condition，然后用这个condition来选择最适合的text。
        # input_features: (batch_size, d_model)
        if type == "img":
            output = self.img_condition_predictor(input_features)
        elif type == "txt":
            output = self.txt_condition_predictor(input_features)
        elif type == "imgtxt":
            output = self.imgtxt_condition_predictor(
                input_features
            )  # (batch_size, d_model * 2)
        else:
            raise ValueError(f"Invalid condition type: {type}")

        # output = output @ self.combiner.label_proj_layer.weight

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
