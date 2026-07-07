"""
Extract CLIP ViT-B/32 features for the 150K RedCaps subsample into the current
shard format (img_features [512], txt_features [512], sample_ids), to a fresh
store. Standalone — replicates the extraction loop in
`src/hook/train_cosir.py::_extract_or_load_features` without the training stack.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.dataset.cosir_datamodule import FeatureExtractionDataset
from src.utils import FeatureManager

# NOTE: we use the raw HF CLIP backbone rather than CoSiRModel on purpose.
# We replicate CoSiRModel.encode_img/encode_txt EXACTLY:
#   img_emb = visual_projection(vision_model(**img).pooler_output)
#   txt_emb = text_projection(text_model(**txt).pooler_output)
# (no normalization). Importing src.model would pull in src.model.clustering ->
# cuml, which has a broken llvmlite in this env. (transformers 5.x also changed
# get_image_features to return the full output object, so we go via submodules.)


def encode_img(model, images):
    out = model.vision_model(**images)
    return model.visual_projection(out.pooler_output)


def encode_txt(model, texts):
    out = model.text_model(**texts)
    return model.text_projection(out.pooler_output)

ANNOT = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"
IMG_ROOT = "/data/PDD"
STORAGE = "/data/SSD2/pre_extract/redcaps_150k/features"
BACKBONE = "openai/clip-vit-base-patch32"
BATCH = 2048
NUM_WORKERS = 8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(os.path.join(STORAGE, "metadata.json")):
        print(f"Store already exists at {STORAGE} — nothing to do.")
        return

    print("Building model + processor …")
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(BACKBONE, use_fast=False)

    dataset = FeatureExtractionDataset(
        annotation_path=ANNOT, image_path=IMG_ROOT, processor=processor, ratio=1
    )
    print(f"Dataset: {len(dataset):,} samples")

    fm = FeatureManager(STORAGE, shard_size=100_000,
                        hdf5_compression=True, hdf5_compression_level=4)

    # Probe feature dims
    probe = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    img_in, txt_in, _ = next(iter(probe))
    img_in = img_in.to(device)
    txt_in = {k: v.to(device) for k, v in txt_in.items()}
    with torch.no_grad():
        img_e = encode_img(model, img_in)
        txt_e = encode_txt(model, txt_in)
    feature_dims = {"img_features": tuple(img_e.shape[1:]),
                    "txt_features": tuple(txt_e.shape[1:])}
    print(f"Feature dims: {feature_dims}")
    del img_in, txt_in, img_e, txt_e, probe

    fm.open_for_writing(len(dataset), feature_dims, backbone_model=BACKBONE)

    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for image_inputs, text_inputs, sample_ids in tqdm(loader, desc="Extracting"):
            sample_ids = [int(s) for s in sample_ids]
            image_inputs = image_inputs.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            img_e = encode_img(model, image_inputs)
            txt_e = encode_txt(model, text_inputs)
            fm.write_batch(img_e, txt_e, sample_ids, img_full=None, txt_full=None)
            torch.cuda.empty_cache()

    fm.finalize_writing()
    print(f"Done. {len(fm.get_all_sample_ids()):,} sample ids written to {STORAGE}")


if __name__ == "__main__":
    main()
