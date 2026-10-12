"""M4: checked CLIP B/32 flip/corner views in M1 painting order.

GPU invocations belong to the controller under /tmp/gpu0.lock. --device cpu
is allowed only with --smoke N. Full outputs are exclusive-create; smoke
outputs use the common smoke writer's overwrite convention.
"""
import r8_common as C

import argparse
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CACHE = Path("/data/SSD2/pre_extract/artelingo_clip224")
FULL_OUT = Path("/data/SSD2/pre_extract/artelingo/r8_clip_views")
VIEW_ORDER = ("horizontal_flip", "top_left", "top_right", "bottom_left", "bottom_right")
CHECK_N = 256
CHECK_THRESHOLD = 0.999   # M4, amendment 1 (was 0.9999)


def load_clip(device):
    """Record and verify the actual Transformers checkpoint before it is read."""
    from transformers import CLIPImageProcessor, CLIPModel, modeling_utils

    records = []
    resolver = modeling_utils._get_resolved_checkpoint_files

    def checked_resolver(*args, **kwargs):
        files, metadata = resolver(*args, **kwargs)
        assert metadata is None and files is not None and len(files) == 1, "unsharded CLIP weights required"
        records.append(C.clip_weight_record(files[0]))
        return files, metadata

    # Pin the rule's main revision and its bin rather than request HF conversion.
    # local_files_only also keeps this research build from downloading new weights.
    with patch.object(modeling_utils, "_get_resolved_checkpoint_files", checked_resolver):
        model = CLIPModel.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            use_safetensors=False, dtype=torch.float32,
        )
    assert len(records) == 1, "Transformers did not resolve exactly one checkpoint"
    processor = CLIPImageProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    assert processor.do_normalize and processor.do_rescale and processor.rescale_factor == 1 / 255
    assert processor.size == {"shortest_edge": 224} and processor.crop_size == {"height": 224, "width": 224}
    model = model.to(device=device, dtype=torch.float32).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    config_dir = Path(records[0]["path"]).parent
    config_hashes = {str(config_dir / name): C.sha_file(config_dir / name)
                     for name in ("config.json", "preprocessor_config.json")}
    return model, mean, std, records[0], config_hashes


def pixels(cache, cache_rows, device):
    a = np.asarray(cache[cache_rows])
    assert a.dtype == np.uint8 and a.shape == (len(cache_rows), 224, 224, 3)
    return torch.from_numpy(a).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255


def five_views(x):
    """Float32 pixels /255; resize the four corners exactly as M4 specifies."""
    assert x.dtype == torch.float32 and x.shape[1:] == (3, 224, 224)
    views = [x.flip(-1)]
    for top, left in ((0, 0), (0, 64), (64, 0), (64, 64)):
        crop = x[:, :, top:top + 160, left:left + 160]
        views.append(F.interpolate(crop, size=(224, 224), mode="bicubic",
                                   align_corners=False, antialias=True).clamp(0, 1))
    return torch.stack(views, dim=1)


def embed(model, x, mean, std):
    with torch.no_grad():
        output = model.get_image_features(pixel_values=(x - mean) / std)
        # Transformers 5 returns BaseModelOutputWithPooling; older releases
        # returned the projected tensor directly.
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        assert features.dtype == torch.float32 and features.shape == (len(x), 512)
        assert torch.isfinite(features).all() and (features.norm(dim=1) > 0).all()
        return F.normalize(features, dim=1).cpu().numpy()


def run(smoke=None, device="cuda", batch_size=128):
    started = perf_counter()
    C.assert_rule()
    C.assert_threads()
    assert batch_size > 0
    assert device in ("cpu", "cuda") and (device != "cpu" or smoke is not None), "CPU is smoke-only"
    assert smoke is None or 0 < smoke <= 36_518
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out = C.SMOKE / "views" if smoke is not None else FULL_OUT
    meta_path, views_path = out / "meta.json", out / "views.npy"
    # M4's fixed external full-output location is the controller's, not an
    # implementer-run destination. Refuse existing full files before any work.
    if smoke is None:
        assert not meta_path.exists() and not views_path.exists(), "views output already exists"
    else:
        # rule gap: M4 fixes 256 check paintings but does not define --smoke N
        # check size. Smoke checks the first min(N, 256); full always checks 256.
        # Smoke arrays never qualify as full M4 outputs (shape and smoke flag).
        assert C.in_smoke(out)
    input_keys = (
        str(CACHE / "images_uint8.npy"), str(CACHE / "paintings.json"),
        "/data/SSD2/pre_extract/artelingo/features/metadata.json",
        "/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json",
        "/data/SSD2/pre_extract/artelingo/style_csd_vitl/row_to_image.npy",
        "/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy",
        "20261117_reader_fix_csd/results/rb_halves.npz",
        "20261031_pseudo_partitions/results/partitions.npz",
    )
    inputs = C.assert_inputs(keys=input_keys, r7=False, clip=True)
    ctx = C.ctx()
    cache_rows = C.painting_cache_row()
    assert cache_rows.dtype == np.int64 and cache_rows.shape == (ctx.n_paintings,)
    assert np.array_equal(ctx.painting_ids, np.asarray(ctx.data.paintings)[ctx.scorer_train[ctx.first_rows]])
    cache = np.load(CACHE / "images_uint8.npy", mmap_mode="r", allow_pickle=False)
    assert cache.dtype == np.uint8 and cache.shape == (49_121, 224, 224, 3)
    assert cache_rows.min() >= 0 and cache_rows.max() < len(cache)
    n = ctx.n_paintings if smoke is None else smoke
    model, mean, std, weight, config_hashes = load_clip(device)
    inputs.update(config_hashes)
    ready = perf_counter()
    check_n = min(n, CHECK_N)
    cached = C.x_paintings()[:check_n]
    assert cached.dtype == np.float32 and cached.shape == (check_n, 512) and np.isfinite(cached).all()
    cosines = []
    for start in range(0, check_n, batch_size):
        stop = min(start + batch_size, check_n)
        features = embed(model, pixels(cache, cache_rows[start:stop], device), mean, std)
        cosines.extend(np.sum(features * cached[start:stop], axis=1).tolist())
    checked = perf_counter()
    passed = bool(np.all(np.asarray(cosines) >= CHECK_THRESHOLD))
    record = {
        "builder": "r8_views.py", "smoke": smoke is not None, "device": device,
        "model": MODEL, "revision": REVISION, "clip_weight": weight,
        "dtype": "float32", "batch_size": batch_size, "batch_unit": "images per model call",
        "eval": True, "no_grad": True, "allow_tf32": False,
        "shape": [n, 5, 512], "unit": True,
        "painting_ids": ctx.painting_ids[:n].tolist(), "view_order": list(VIEW_ORDER),
        "preprocessing": {"mean": mean.flatten().cpu().tolist(), "std": std.flatten().cpu().tolist(),
                          "pixels": "float32 /255", "crop_size": 160, "resize_size": 224,
                          "resize_mode": "bicubic", "align_corners": False,
                          "antialias": True, "clamp": [0, 1]},
        "joins": {"painting_order": "M1 g_st / rb_halves paintings_sorted",
                  "cache_row": "r8_common.painting_cache_row() via paintings.json painting id",
                  "cache_rows": cache_rows[:n].tolist(),
                  "first_scorer_train_global_rows": ctx.scorer_train[ctx.first_rows[:n]].tolist(),
                  "painting_ids_equal_data_first_rows": True},
        "input_sha256": inputs, "views_sha256": None,
        "view_check": {"passed": passed, "n": check_n, "threshold": CHECK_THRESHOLD,
                       "painting_indices": list(range(check_n)), "cosines": cosines,
                       "min_cosine": min(cosines)},
        "timing_seconds": {"setup": ready - started, "check": checked - ready},
    }
    out.mkdir(parents=True, exist_ok=True)
    if not passed:
        record["dropped"] = ["V100", "V67", "V33"]
        record["timing_seconds"]["total"] = perf_counter() - started
        C.write_json_once(meta_path, record)
        raise SystemExit("M4 view check failed; recorded in meta.json; no views written")

    # One view per call keeps the default 128-image batch bounded on a 24 GB
    # card; output axes remain painting, view, feature.
    views = np.empty((n, 5, 512), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = five_views(pixels(cache, cache_rows[start:stop], device))
        for v in range(5):
            views[start:stop, v] = embed(model, batch[:, v], mean, std)
    assert np.isfinite(views).all() and np.allclose(np.linalg.norm(views, axis=-1), 1, atol=1e-5)
    extracted = perf_counter()
    with open(views_path, "wb" if smoke is not None else "xb") as f:
        np.save(f, views, allow_pickle=False)
    record["views_sha256"] = C.sha_file(views_path)
    record["timing_seconds"].update(views=extracted - checked, write_hash=perf_counter() - extracted,
                                    total=perf_counter() - started)
    C.write_json_once(meta_path, record)
    print(f"M4 views: {n} paintings; check min cosine {min(cosines):.7f}; {views_path}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, metavar="N", help="first N paintings, results/smoke/views only")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cpu requires --smoke")
    parser.add_argument("--batch-size", type=int, default=128, help="images per CLIP call (default: 128 for 24 GB)")
    args = parser.parse_args()
    run(args.smoke, args.device, args.batch_size)


if __name__ == "__main__":
    main()
