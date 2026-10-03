import numpy as np
from PIL import Image

from src.data.feature_extract import load_encoder


def test_clip_encoder_shapes_and_norm():
    enc = load_encoder("clip_b32", device="cpu")
    img = enc.encode_images([Image.new("RGB", (64, 64), (255, 0, 0))] * 2, batch_size=2)
    txt = enc.encode_texts(["a red square", "a blue circle"], batch_size=2)
    assert img.shape == (2, 512) and txt.shape == (2, 512) and enc.dim == 512
    assert np.allclose(np.linalg.norm(img, axis=1), 1, atol=1e-4)
    assert float(img[0] @ txt[0]) > float(img[0] @ txt[1])
