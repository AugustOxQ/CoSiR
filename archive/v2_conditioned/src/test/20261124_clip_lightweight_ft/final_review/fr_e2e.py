"""Final review: end-to-end spot check of the three selected models on CPU, with own model assembly and preprocessing.

Loads each selected run's best_params.pt into a fresh CLIP (LB: by name; LoRA: own peft injection with the spec's
config; LP: the two maps applied to the frozen features), encodes a handful of selection rows (images from the uint8
cache, captions from annotations[sample_ids[row]]) and compares with the run's features.npz rows. Also checks the
untrained cache path against features_epoch0.npz.

Run (CPU): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  /root/miniconda3/envs/CoSiR/bin/python fr_e2e.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402

JOBS = ROOT / "res/cluster_jobs"
CACHE = Path("/data/SSD2/pre_extract/artelingo_clip224")
CLIP = "openai/clip-vit-base-patch32"
SEL = {"LP": ("20261007-070950-65bb2f4", "LP_lr3e-4", 9), "LB": ("20261007-071151-65bb2f4", "LB_lr3e-5", 8),
       "LoRA": ("20261007-071251-65bb2f4", "LoRA_lr1e-4", 10)}
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def fresh_clip():
    from transformers import CLIPModel
    return CLIPModel.from_pretrained(CLIP).float().eval()


def compare(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
    return {"max_abs": float((a - b).abs().max()), "max_rel": float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max()),
            "min_cos": float(F.cosine_similarity(a, b, dim=-1).min())}


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    data = load_artelingo()
    sp = artelingo_splits(data)
    ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
    sids, paint = np.asarray(data.sample_ids), np.asarray(data.paintings)
    pj = json.loads((CACHE / "paintings.json").read_text())["paintings"]
    images = np.load(CACHE / "images_uint8.npy", mmap_mode="r")
    rng = np.random.default_rng(7)
    rows = np.sort(rng.choice(np.asarray(sp.selection), 12, replace=False))
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP)
    enc = tok([ann[int(sids[r])]["caption"] for r in rows], return_tensors="pt", max_length=77,
              padding="max_length", truncation=True)
    u8 = torch.from_numpy(np.stack([images[pj[str(paint[r])]["index"]] for r in rows]))
    pix = (u8.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD

    def encode(m):
        with torch.no_grad():
            i = m.visual_projection(m.vision_model(pixel_values=pix).pooler_output)
            t = m.text_projection(m.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output)
        return i, t

    def stored(tag, name, f):
        z = np.load(JOBS / tag / "code/outputs/clipft" / name / f)
        pos = np.searchsorted(z["rows"], rows)
        assert np.array_equal(z["rows"][pos], rows)
        return z["img"][pos], z["txt"][pos]

    out = {"rows": rows.tolist()}
    # untrained cache path vs features_epoch0.npz of the selected LB run
    base = fresh_clip()
    i0, t0 = encode(base)
    si, st = stored(*SEL["LB"][:2], "features_epoch0.npz")
    out["epoch0_cache_path"] = {"img": compare(i0, si), "txt": compare(t0, st)}

    # LB: overwrite the trained tensors by name
    tag, name, ep = SEL["LB"]
    ck = torch.load(JOBS / tag / "code/outputs/clipft" / name / "best_params.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == ep and ck["variant"] == "LB"
    m = fresh_clip()
    params = dict(m.named_parameters())
    keys = set(ck["params"])
    allowed = ("vision_model.encoder.layers.11.", "text_model.encoder.layers.11.", "vision_model.post_layernorm.",
               "text_model.final_layer_norm.", "visual_projection.", "text_projection.")
    expect = {n for n in params if n == "logit_scale" or n.startswith(allowed)}
    out["LB_saved_keys_equal_spec_set"] = keys == expect
    out["LB_n_saved_params"] = int(sum(v.numel() for v in ck["params"].values()))
    with torch.no_grad():
        for n, v in ck["params"].items():
            params[n].copy_(v)
    i, t = encode(m)
    si, st = stored(tag, name, "features.npz")
    out["LB"] = {"img": compare(i, si), "txt": compare(t, st), "img_vs_epoch0": compare(i, i0)}

    # LoRA: own injection with the spec's config, then load adapters and logit_scale
    tag, name, ep = SEL["LoRA"]
    ck = torch.load(JOBS / tag / "code/outputs/clipft" / name / "best_params.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == ep and ck["variant"] == "LoRA"
    from peft import LoraConfig, inject_adapter_in_model
    m = fresh_clip()
    m = inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]), m)
    params = dict(m.named_parameters())
    lora_names = {n for n in params if "lora_" in n}
    out["LoRA_saved_keys"] = {"n": len(ck["params"]), "missing_in_model": sorted(set(ck["params"]) - set(params))[:5],
                              "equal_lora_plus_logit_scale": set(ck["params"]) == lora_names | {"logit_scale"}}
    out["LoRA_n_saved_params"] = int(sum(v.numel() for v in ck["params"].values()))
    with torch.no_grad():
        for n, v in ck["params"].items():
            params[n].copy_(v)
    m.eval()
    i, t = encode(m)
    si, st = stored(tag, name, "features.npz")
    out["LoRA"] = {"img": compare(i, si), "txt": compare(t, st), "img_vs_epoch0": compare(i, i0)}

    # LP: maps on the frozen cached features
    tag, name, ep = SEL["LP"]
    ck = torch.load(JOBS / tag / "code/outputs/clipft" / name / "best_params.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == ep and ck["variant"] == "LP"
    out["LP_saved_keys"] = sorted(ck["params"])
    Wi, Wt = ck["params"]["img_map.weight"], ck["params"]["txt_map.weight"]
    i = torch.from_numpy(data.img_features[rows]) @ Wi.T
    t = torch.from_numpy(data.txt_features[rows]) @ Wt.T
    si, st = stored(tag, name, "features.npz")
    out["LP"] = {"img": compare(i, si), "txt": compare(t, st),
                 "map_dist_from_identity": [float((Wi - torch.eye(512)).norm()), float((Wt - torch.eye(512)).norm())]}
    print(json.dumps(out, indent=1))
    (HERE / "fr_e2e.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
