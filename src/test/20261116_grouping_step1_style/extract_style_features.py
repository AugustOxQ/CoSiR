"""Step 1 style features (PLAN.md §1): CSD ViT-L style embeddings and VGG-19 Gram descriptors, one per painting image.

Subcommands
  download  fetch the CSD ViT-L checkpoint (Hugging Face, pinned revision) and torchvision VGG-19 IMAGENET1K_V1 weights
            into /data/SSD2 and verify their SHA-256 (needs network; nothing else needs it).
  smoke     both extractors on the first 256 images of the image order, written to <out>/smoke/ (overwritten each run;
            the Gram PCA is fitted on those 256 images as a stand-in for the 8,000-image fit sample).
  full      all 61,402 images. Refuses to overwrite existing outputs.

Image order: the distinct `image` paths of /data/PDD/artelingo/artelingo_train.json in order of first appearance (the
order of src/test/20260929_cross_encoder_stage1/extract_features.py, whose DINOv2 rows are rebuilt from it here as a
check). Outputs per folder (`style_csd_vitl/`, `style_vgg19_gram/` under /data/SSD2/pre_extract/artelingo/):
  embeddings.npy                     float32 (61402, d), row j = image j, L2-normalised
  row_to_image.npy                   int32 (308723,), row r of src.data.artelingo.load_artelingo() -> image index
  row_to_image_annotation_order.npy  int32 (308723,), row i of artelingo_train.json -> image index
  meta.json                          model, weights SHA-256, preprocessing, image order, alignment checks, PLAN SHA-256
  gram_pca.npz (Gram only)           per-layer PCA mean_, components_, explained_variance_ratio_, fit image indices
Row-aligned features for load_artelingo() order: np.load("embeddings.npy")[np.load("row_to_image.npy")].

Run from the repository root with the CoSiR env, under the GPU lock (PLAN.md §7):
  flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python -u \
      src/test/20261116_grouping_step1_style/extract_style_features.py {smoke|full}
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import VGG19_Weights

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from csd_model import VIT_L14, CSDStyleModel  # noqa: E402

PLAN_PATH = HERE / "PLAN.md"
PLAN_SHA256 = "b00a4a7aa796957addc232dd92e1b1579306f8b312748a36ac1479331f8b0520"

ANNOTATIONS = Path("/data/PDD/artelingo/artelingo_train.json")
IMAGE_ROOT = Path("/data/PDD/wikiart_proj/wikiart")
EXPECTED_ROWS = 308_723
EXPECTED_IMAGES = 61_402
DINOV2_CHECK = REPO_ROOT / "src/test/20260929_cross_encoder_stage1/features/dinov2_img.npy"

OUT_ROOT = Path("/data/SSD2/pre_extract/artelingo")
CSD_DIR = OUT_ROOT / "style_csd_vitl"
GRAM_DIR = OUT_ROOT / "style_vgg19_gram"

# CSD: the authors' release, linked from https://github.com/learn2phoenix/CSD (README, "CSD Model (ViT-L)").
CSD_GITHUB = "https://github.com/learn2phoenix/CSD"
CSD_REPO_ID = "tomg-group-umd/CSD-ViT-L"
CSD_REVISION = "c94c4d51863d380511e561404744b848059665ca"  # commit that added pytorch_model.bin
CSD_FILENAME = "pytorch_model.bin"
CSD_SHA256 = "40e92fad63a361b8136100cd234c42d401ef9b34ff1748234318929ebcc7e7a1"  # as listed by Hugging Face (LFS)
CSD_BYTES = 2_438_228_893
HF_CACHE = Path("/data/SSD2/HF_home/hub")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)  # CSD/loss_utils.py transforms_branch0
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# VGG-19: torchvision IMAGENET1K_V1, cached under the Gram folder (torch hub dir) instead of ~/.cache/torch.
VGG_WEIGHTS = VGG19_Weights.IMAGENET1K_V1
VGG_HUB_DIR = GRAM_DIR / "weights" / "torch_hub"
VGG_FILE = VGG_HUB_DIR / "checkpoints" / Path(VGG_WEIGHTS.url).name
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VGG_SIZE = 384
# torchvision vgg19().features indices of relu1_1 ... relu5_1, with the conv feeding each and its channel count.
RELU_LAYERS = OrderedDict([("relu1_1", (1, 64)), ("relu2_1", (6, 128)), ("relu3_1", (11, 256)),
                           ("relu4_1", (20, 512)), ("relu5_1", (29, 512))])
PCA_DIM = 128
PCA_FIT_PAINTINGS = 8_000
PCA_SAMPLE_SEED = 0
PCA_SOLVER_SEED = 0
EXPECTED_SCORER_GROUPS = 36_518

SMOKE_IMAGES = 256
CHECK_ROWS = 512        # fit images re-checked end to end (pass 1 Gram -> projection vs pass 2 output)
SPOT_IMAGES = 8         # images re-encoded one at a time after the pass (row bookkeeping check)

CSD_TRANSFORM = T.Compose([
    T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])
VGG_TRANSFORM = T.Compose([
    T.Resize(VGG_SIZE),  # shorter side -> 384, torchvision default bilinear (PIL, antialiased)
    T.CenterCrop(VGG_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
CSD_PREPROCESSING = ("PIL convert('RGB'); Resize(224, BICUBIC) on the shorter side; CenterCrop(224); ToTensor; "
                     f"Normalize(mean={CLIP_MEAN}, std={CLIP_STD}) -- CSD/loss_utils.py transforms_branch0")
VGG_PREPROCESSING = ("PIL convert('RGB'); Resize(384) on the shorter side (torchvision default BILINEAR, PIL "
                     f"antialiased); CenterCrop(384); ToTensor; Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})")


def log(message: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assert_plan() -> None:
    got = sha256_file(PLAN_PATH)
    if got != PLAN_SHA256:
        raise AssertionError(f"PLAN.md SHA-256 {got} differs from the fixed design {PLAN_SHA256}")


# --------------------------------------------------------------------------------------------------------------------
# Image order and row alignment
# --------------------------------------------------------------------------------------------------------------------

def load_image_table() -> tuple[list[dict], list[str], np.ndarray]:
    """Annotation rows, distinct image paths in first-appearance order, annotation row -> image index."""
    with ANNOTATIONS.open() as file:
        annotations = json.load(file)
    if len(annotations) != EXPECTED_ROWS:
        raise AssertionError(f"expected {EXPECTED_ROWS:,} annotation rows, got {len(annotations):,}")
    paths = [row["image"] for row in annotations]
    image_paths = list(dict.fromkeys(paths))
    if len(image_paths) != EXPECTED_IMAGES:
        raise AssertionError(f"expected {EXPECTED_IMAGES:,} distinct images, got {len(image_paths):,}")
    index = {path: j for j, path in enumerate(image_paths)}
    row_to_image_ann = np.fromiter((index[p] for p in paths), dtype=np.int32, count=len(paths))
    return annotations, image_paths, row_to_image_ann


def first_row_per_image(row_to_image: np.ndarray, n_images: int) -> np.ndarray:
    first = np.full(n_images, -1, dtype=np.int64)
    order = np.argsort(row_to_image, kind="stable")
    sorted_images = row_to_image[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_images)) + 1]
    first[sorted_images[starts]] = order[starts]
    if (first < 0).any():
        raise AssertionError("some image has no annotation row")
    return first


def clip_mapping_max_diff(img_features: np.ndarray, row_to_image: np.ndarray, n_images: int) -> float:
    """Max |CLIP(row) - CLIP(first row of the same image)|; rows of one image share a CLIP vector if aligned."""
    rep = first_row_per_image(row_to_image, n_images)[row_to_image]
    worst = 0.0
    for start in range(0, len(row_to_image), 32_768):
        stop = start + 32_768
        diff = np.abs(img_features[start:stop] - img_features[rep[start:stop]]).max()
        worst = max(worst, float(diff))
    return worst


def align_rows(annotations: list[dict], image_paths: list[str], row_to_image_ann: np.ndarray):
    """row_to_image in load_artelingo() order, with assertion checks against that loader's own CLIP image features."""
    from src.data.artelingo import load_artelingo

    data = load_artelingo()
    sample_ids = np.asarray(data.sample_ids, dtype=np.int64)
    row_to_image = row_to_image_ann[sample_ids].astype(np.int32)
    n = len(image_paths)
    checks = {"load_artelingo_rows": int(len(sample_ids)),
              "sample_ids_identity_order": bool(np.array_equal(sample_ids, np.arange(len(sample_ids)))),
              "sample_ids_sha256": sha256_array(sample_ids)}

    # (1) Every image is used; image <-> painting is one-to-one; load_artelingo's painting per row matches.
    counts = np.bincount(row_to_image, minlength=n)
    assert counts.min() >= 1 and len(counts) == n, "an image index has no row"
    painting_of_image = {}
    for row, j in zip(annotations, row_to_image_ann.tolist()):
        if painting_of_image.setdefault(j, row["painting"]) != row["painting"]:
            raise AssertionError(f"image {image_paths[j]} carries more than one painting")
    assert len(set(painting_of_image.values())) == n, "a painting has more than one image"
    paintings = np.asarray([painting_of_image[j] for j in range(n)], dtype=object)
    assert np.array_equal(np.asarray(data.paintings, dtype=object), paintings[row_to_image]), \
        "load_artelingo() paintings disagree with row_to_image"
    checks["rows_per_image_min_max"] = [int(counts.min()), int(counts.max())]
    checks["image_painting_one_to_one"] = True

    # (2) load_artelingo()'s CLIP image vectors: rows mapped to one image carry the same vector (up to GPU float noise
    # of the original extraction); distinct images carry distinct vectors; a one-row shift breaks the check.
    img = data.img_features
    diff = clip_mapping_max_diff(img, row_to_image, n)
    assert diff < 1e-4, f"rows mapped to one image have different CLIP vectors (max abs diff {diff})"
    reps = first_row_per_image(row_to_image, n)
    distinct = len({hashlib.md5(np.ascontiguousarray(img[r]).tobytes()).hexdigest() for r in reps.tolist()})
    assert distinct == n, f"{n - distinct} images share a CLIP vector with another image"
    shifted = clip_mapping_max_diff(img, np.roll(row_to_image, 1), n)
    assert shifted > 1e-2, f"negative control: a one-row shift should break the CLIP check (got {shifted})"
    checks["clip_same_image_max_abs_diff"] = diff
    checks["clip_distinct_vectors_over_images"] = [distinct, n]
    checks["clip_shifted_mapping_max_abs_diff"] = shifted

    # (3) DINOv2 rows (annotation order, built by the stage-1 script from the same unique-path order) are constant
    # within each image index of row_to_image_annotation_order.
    if DINOV2_CHECK.exists():
        dino = np.load(DINOV2_CHECK, mmap_mode="r")
        assert dino.shape[0] == EXPECTED_ROWS
        rep_ann = first_row_per_image(row_to_image_ann, n)[row_to_image_ann]
        equal = 0
        for start in range(0, EXPECTED_ROWS, 32_768):
            block = np.asarray(dino[start:start + 32_768])
            equal += int(np.all(block == np.asarray(dino[rep_ann[start:start + 32_768]]), axis=1).sum())
        assert equal == EXPECTED_ROWS, f"only {equal} DINOv2 rows equal their image's first row"
        checks["dinov2_rows_equal_image_rep"] = [equal, EXPECTED_ROWS]
        checks["dinov2_alignment_note"] = ("dinov2_img.npy row i = artelingo_train.json row i; in load_artelingo() "
                                           "order it is dinov2_img[data.sample_ids], i.e. the same positional join "
                                           "as embeddings[row_to_image]")
    else:
        checks["dinov2_rows_equal_image_rep"] = "skipped (file missing)"
    return data, row_to_image, checks


def pca_fit_images(data, row_to_image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Images of 8,000 scorer-train paintings (painting = split group), drawn with default_rng(0)."""
    from src.data.artelingo_splits import artelingo_splits

    splits = artelingo_splits(data)
    scorer_rows = splits.scorer_train
    scorer_groups = np.unique(splits.groups[scorer_rows])
    if len(scorer_groups) != EXPECTED_SCORER_GROUPS:
        raise AssertionError(f"expected {EXPECTED_SCORER_GROUPS} scorer-train groups, got {len(scorer_groups)}")
    chosen = np.random.default_rng(PCA_SAMPLE_SEED).choice(scorer_groups, size=PCA_FIT_PAINTINGS, replace=False)
    rows = scorer_rows[np.isin(splits.groups[scorer_rows], chosen)]
    images = np.unique(row_to_image[rows]).astype(np.int64)
    if len(images) != PCA_FIT_PAINTINGS:
        raise AssertionError(f"{PCA_FIT_PAINTINGS} groups should hold {PCA_FIT_PAINTINGS} images, got {len(images)}")
    info = {"rule": ("groups = artelingo_splits(load_artelingo()).groups; G = np.unique(groups[scorer_train]); "
                     "chosen = np.random.default_rng(0).choice(G, size=8000, replace=False); fit images = "
                     "np.unique(row_to_image[scorer_train rows whose group is in chosen]) (sorted)"),
            "scorer_train_rows": int(len(scorer_rows)), "scorer_train_groups": int(len(scorer_groups)),
            "chosen_groups": int(len(chosen)), "fit_rows": int(len(rows)), "fit_images": int(len(images)),
            "fit_image_indices_sha256": sha256_array(images)}
    return images, info


# --------------------------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------------------------

def csd_weight_path() -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(CSD_REPO_ID, CSD_FILENAME, revision=CSD_REVISION, cache_dir=str(HF_CACHE),
                                    local_files_only=True))
    except Exception as error:  # noqa: BLE001 -- any cache miss means the download step has not run
        raise FileNotFoundError(f"CSD checkpoint not in {HF_CACHE}; run the `download` subcommand first") from error


def load_csd(device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = csd_weight_path()
    digest = sha256_file(path)
    if digest != CSD_SHA256:
        raise AssertionError(f"CSD checkpoint SHA-256 {digest} differs from the official {CSD_SHA256}")
    # The file is a training checkpoint pickle (argparse.Namespace, numpy scalars), so weights_only=True cannot
    # unpickle it; it is loaded only after its SHA-256 matched the authors' release above.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    top_keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    state = checkpoint["model_state_dict"] if "model_state_dict" in top_keys else checkpoint
    state = OrderedDict((k.replace("module.", "") if k.startswith("module.") else k, v)  # CSD convert_state_dict
                        for k, v in state.items())
    model = CSDStyleModel()
    message = model.load_state_dict(state, strict=False)  # as in CSD main_sim.py; then asserted complete
    if message.missing_keys or message.unexpected_keys:
        raise AssertionError(f"CSD state dict mismatch: missing {message.missing_keys}, "
                             f"unexpected {message.unexpected_keys}")
    model = model.to(device).eval()
    info = {"source": f"https://huggingface.co/{CSD_REPO_ID}/blob/{CSD_REVISION}/{CSD_FILENAME}",
            "official_repository": CSD_GITHUB, "paper": "Somepalli et al. 2024, arXiv 2404.01292",
            "hf_repo": CSD_REPO_ID, "hf_revision": CSD_REVISION, "file": str(path),
            "file_bytes": path.stat().st_size, "file_sha256": digest,
            "licence": "code MIT (GitHub LICENSE); Hugging Face model card: cc-by-4.0",
            "checkpoint_top_level_keys": top_keys,
            "checkpoint_epoch": (int(checkpoint["epoch"]) if "epoch" in top_keys else None),
            "state_dict_tensors": len(state), "load_state_dict": "strict=False, asserted 0 missing / 0 unexpected",
            "architecture": f"CSD_CLIP('vit_large', content_proj_head='default'); OpenAI CLIP ViT-L/14 visual {VIT_L14}",
            "vendored_code": "csd_model.py (OpenAI CLIP VisionTransformer + CSD_CLIP forward, MIT)",
            "embedding": ("style head: normalize(backbone(x) @ last_layer_style), 768-d (CSD main_sim.py "
                          "eval_embed='head' default)"),
            "precision": ("torch.autocast('cuda', float16) as CSD's extract_features(use_fp16=True); output cast "
                          "to float32 and L2-renormalised")}
    return model, info


def load_vgg(device: torch.device) -> tuple[torch.nn.Module, dict]:
    if not VGG_FILE.exists():
        raise FileNotFoundError(f"{VGG_FILE} missing; run the `download` subcommand first")
    torch.hub.set_dir(str(VGG_HUB_DIR))  # vgg19(weights=...) then resolves to the cached file, no download
    model = torchvision.models.vgg19(weights=VGG_WEIGHTS)
    features = model.features[:RELU_LAYERS["relu5_1"][0] + 1]
    for name, (index, channels) in RELU_LAYERS.items():
        conv = features[index - 1]
        assert isinstance(features[index], torch.nn.ReLU), name
        assert isinstance(conv, torch.nn.Conv2d) and conv.out_channels == channels, name
    features = features.to(device).eval()
    digest = sha256_file(VGG_FILE)
    prefix = VGG_FILE.stem.split("-")[-1]
    assert digest.startswith(prefix), f"VGG weights SHA-256 {digest} does not start with torchvision's {prefix}"
    info = {"model": "torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30]",
            "url": VGG_WEIGHTS.url, "file": str(VGG_FILE), "file_sha256": digest,
            "torchvision": torchvision.__version__}
    return features, info


# --------------------------------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------------------------------

class ImageSet(Dataset):
    def __init__(self, image_paths: list[str], indices: np.ndarray, csd: bool, vgg: bool):
        self.paths, self.indices, self.csd, self.vgg = image_paths, np.asarray(indices, dtype=np.int64), csd, vgg

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        j = int(self.indices[k])
        with Image.open(IMAGE_ROOT / self.paths[j]) as image:
            rgb = image.convert("RGB")
        csd = CSD_TRANSFORM(rgb) if self.csd else torch.empty(0)
        vgg = VGG_TRANSFORM(rgb) if self.vgg else torch.empty(0)
        return j, csd, vgg


def make_loader(image_paths, indices, csd: bool, vgg: bool, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(ImageSet(image_paths, indices, csd, vgg), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True, prefetch_factor=4 if workers else None)


@torch.inference_mode()
def csd_embed(model, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16):
        _, _, style = model(x)
    return F.normalize(style.float(), dim=1)


TRIU_CACHE: dict = {}


def triu(channels: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (channels, str(device))
    if key not in TRIU_CACHE:
        rows, cols = torch.triu_indices(channels, channels, device=device)  # row-major, i <= j, diagonal included
        TRIU_CACHE[key] = (rows, cols)
    return TRIU_CACHE[key]


@torch.inference_mode()
def gram_upper(features, x: torch.Tensor) -> list[torch.Tensor]:
    """Per layer relu1_1..relu5_1: G = F F^T / (H W), upper triangle with diagonal, float32."""
    wanted = {index for index, _ in RELU_LAYERS.values()}
    out, h = [], x
    for index, layer in enumerate(features):
        h = layer(h)
        if index in wanted:
            b, c, height, width = h.shape
            flat = h.reshape(b, c, height * width)
            gram = torch.bmm(flat, flat.transpose(1, 2)) / (height * width)
            rows, cols = triu(c, h.device)
            out.append(gram[:, rows, cols])
    return out


class GramProjector:
    """Per-layer PCA projection on the GPU, block L2-normalisation, concatenation, final L2-normalisation."""

    def __init__(self, pcas: list, device):
        self.means = [torch.as_tensor(p.mean_, dtype=torch.float32, device=device) for p in pcas]
        self.components = [torch.as_tensor(p.components_, dtype=torch.float32, device=device) for p in pcas]

    @torch.inference_mode()
    def blocks(self, grams: list[torch.Tensor]) -> list[torch.Tensor]:
        return [(g - m) @ c.T for g, m, c in zip(grams, self.means, self.components)]

    @torch.inference_mode()
    def __call__(self, grams: list[torch.Tensor]) -> torch.Tensor:
        blocks = [F.normalize(b, dim=1) for b in self.blocks(grams)]
        return F.normalize(torch.cat(blocks, dim=1), dim=1)


def fit_gram_pca(features, image_paths, fit_images: np.ndarray, device, batch_size: int, workers: int):
    """Pass 1: Gram upper triangles of the fit images (kept on the host), then a randomized PCA per layer."""
    from sklearn.decomposition import PCA

    dims = [c * (c + 1) // 2 for _, c in RELU_LAYERS.values()]
    stored = [np.empty((len(fit_images), d), dtype=np.float32) for d in dims]
    log(f"Gram pass 1: {len(fit_images):,} fit images, dims {dims}, host memory "
        f"{sum(d * len(fit_images) * 4 for d in dims) / 2**30:.2f} GiB")
    started, done = perf_counter(), 0
    position = {int(j): k for k, j in enumerate(fit_images)}
    for indices, _, vgg in make_loader(image_paths, fit_images, False, True, batch_size, workers):
        grams = gram_upper(features, vgg.to(device, non_blocking=True))
        where = np.asarray([position[int(j)] for j in indices])
        for layer, g in enumerate(grams):
            stored[layer][where] = g.cpu().numpy()
        done += len(indices)
        if done % (batch_size * 20) < batch_size or done == len(fit_images):
            log(f"  pass 1 {done:,}/{len(fit_images):,} ({done / (perf_counter() - started):.1f} img/s)")
    pass1_seconds = perf_counter() - started
    check = [s[:CHECK_ROWS].copy() for s in stored]
    pcas, layer_info, fit_started = [], [], perf_counter()
    for (name, (_, channels)), layer in zip(RELU_LAYERS.items(), range(len(stored))):
        t0 = perf_counter()
        pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=PCA_SOLVER_SEED, copy=False)
        pca.fit(stored[layer])
        stored[layer] = None  # free the centred copy before the next layer
        assert np.isfinite(pca.components_).all() and np.isfinite(pca.mean_).all(), name
        layer_info.append({"layer": name, "channels": channels, "gram_dim": channels * (channels + 1) // 2,
                           "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                           "fit_seconds": round(perf_counter() - t0, 3)})
        log(f"  PCA {name}: {layer_info[-1]['gram_dim']} -> {PCA_DIM}, explained "
            f"{layer_info[-1]['explained_variance_ratio_sum']:.4f}, {layer_info[-1]['fit_seconds']} s")
        pcas.append(pca)
    timing = {"pass1_seconds": round(pass1_seconds, 3), "pca_fit_seconds": round(perf_counter() - fit_started, 3)}
    return pcas, check, layer_info, timing


def gpu_vs_sklearn(projector: GramProjector, pcas, check: list[np.ndarray], device) -> float:
    """Max relative difference between the GPU projection and sklearn PCA.transform on the check rows."""
    grams = [torch.as_tensor(c, device=device) for c in check]
    worst = 0.0
    for block, pca, raw in zip(projector.blocks(grams), pcas, check):
        reference = pca.transform(raw)
        scale = np.abs(reference).max()
        worst = max(worst, float(np.abs(block.cpu().numpy() - reference).max() / scale))
    return worst


def extract_all(image_paths, indices, csd_model, vgg_features, projector, device, batch_size, workers):
    """Pass 2: CSD style and/or Gram descriptors for `indices`; row k of each output = image indices[k]."""
    n = len(indices)
    csd_out = np.empty((n, VIT_L14["output_dim"]), dtype=np.float32) if csd_model is not None else None
    gram_out = np.empty((n, PCA_DIM * len(RELU_LAYERS)), dtype=np.float32) if projector is not None else None
    position = {int(j): k for k, j in enumerate(indices)}
    started, done = perf_counter(), 0
    loader = make_loader(image_paths, indices, csd_model is not None, projector is not None, batch_size, workers)
    for batch_indices, csd_x, vgg_x in loader:
        where = np.asarray([position[int(j)] for j in batch_indices])
        if csd_model is not None:
            csd_out[where] = csd_embed(csd_model, csd_x.to(device, non_blocking=True)).cpu().numpy()
        if projector is not None:
            gram_out[where] = projector(gram_upper(vgg_features, vgg_x.to(device, non_blocking=True))).cpu().numpy()
        done += len(batch_indices)
        if done % (batch_size * 50) < batch_size or done == n:
            log(f"  pass 2 {done:,}/{n:,} ({done / (perf_counter() - started):.1f} img/s)")
    return csd_out, gram_out, perf_counter() - started


def spot_check(image_paths, indices, outputs: np.ndarray, encode, want_csd: bool, rng_seed: int = 0) -> float:
    """Re-encode a few images one at a time; min cosine with their stored rows (row bookkeeping check)."""
    picks = sorted({0, len(indices) - 1, *np.random.default_rng(rng_seed).choice(len(indices), SPOT_IMAGES - 2,
                                                                                    replace=False).tolist()})
    dataset = ImageSet(image_paths, np.asarray(indices)[picks], want_csd, not want_csd)
    cosines = []
    for k, row in enumerate(picks):
        _, csd_x, vgg_x = dataset[k]
        vector = encode((csd_x if want_csd else vgg_x).unsqueeze(0)).cpu().numpy()[0]
        cosines.append(float(vector @ outputs[row]))
    return min(cosines)


def summarize(name: str, array: np.ndarray) -> dict:
    if not np.isfinite(array).all():
        raise AssertionError(f"{name}: non-finite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if norms.min() < 0.9999 or norms.max() > 1.0001:
        raise AssertionError(f"{name}: row norms outside [0.9999, 1.0001]: {norms.min()}, {norms.max()}")
    stats = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": True,
             "norm_min_mean_max": [float(norms.min()), float(norms.mean()), float(norms.max())],
             "value_min_max": [float(array.min()), float(array.max())],
             "distinct_rows": int(len(np.unique(array.round(6), axis=0)))}
    log(f"{name}: {stats}")
    return stats


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
            "gpu": torch.cuda.get_device_name(0), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "hostname": platform.node()}


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def run(mode: str, which: str, batch_size: int, workers: int) -> None:
    assert_plan()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = False  # full float32 Gram, PCA projection and VGG convolutions
    torch.backends.cudnn.allow_tf32 = False
    want_csd, want_gram = which in ("csd", "both"), which in ("gram", "both")
    smoke = mode == "smoke"
    csd_dir = CSD_DIR / "smoke" if smoke else CSD_DIR
    gram_dir = GRAM_DIR / "smoke" if smoke else GRAM_DIR
    targets = []
    if want_csd:
        targets += [csd_dir / name for name in ("embeddings.npy", "meta.json")]
    if want_gram:
        targets += [gram_dir / name for name in ("embeddings.npy", "meta.json", "gram_pca.npz")]
    if not smoke:
        targets += [d / name for d in ([csd_dir] if want_csd else []) + ([gram_dir] if want_gram else [])
                    for name in ("row_to_image.npy", "row_to_image_annotation_order.npy")]
        existing = [str(t) for t in targets if t.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing results: {existing}")
    for d in {t.parent for t in targets}:
        d.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    device = torch.device("cuda")
    log(f"mode={mode} which={which} batch={batch_size} workers={workers}")
    annotations, image_paths, row_to_image_ann = load_image_table()
    data, row_to_image, alignment = align_rows(annotations, image_paths, row_to_image_ann)
    log(f"alignment checks passed: {alignment}")
    indices = np.arange(SMOKE_IMAGES if smoke else len(image_paths), dtype=np.int64)
    common = {
        "plan": {"path": str(PLAN_PATH.relative_to(REPO_ROOT)), "sha256": PLAN_SHA256, "section": "§1 Features"},
        "script": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                   "sha256": sha256_file(Path(__file__).resolve()),
                   "csd_model_py_sha256": sha256_file(HERE / "csd_model.py")},
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode, "environment": environment(),
        "images": {"root": str(IMAGE_ROOT), "annotations": str(ANNOTATIONS),
                   "annotations_sha256": sha256_file(ANNOTATIONS),
                   "order": ("distinct annotation `image` paths in order of first appearance in artelingo_train.json "
                             "(as src/test/20260929_cross_encoder_stage1/extract_features.py); row j of "
                             "embeddings.npy = image_paths[j]"),
                   "n_images_total": len(image_paths), "n_images_in_this_file": int(len(indices)),
                   "image_paths": image_paths if not smoke else image_paths[:SMOKE_IMAGES]},
        "row_mapping": {
            "row_to_image.npy": ("int32 (308723,): row r of src.data.artelingo.load_artelingo() -> image index; "
                                 "= row_to_image_annotation_order[data.sample_ids]"),
            "row_to_image_annotation_order.npy": "int32 (308723,): row i of artelingo_train.json -> image index",
            "usage": "features_in_load_artelingo_order = embeddings[row_to_image]",
            "row_to_image_sha256": sha256_array(row_to_image),
            "row_to_image_annotation_order_sha256": sha256_array(row_to_image_ann),
            "checks": alignment} if not smoke else f"full run only (smoke rows = images 0..{SMOKE_IMAGES - 1})",
    }

    csd_model = csd_info = vgg_features = vgg_info = projector = None
    pcas, gram_extra, timing = None, {}, {}
    if want_csd:
        csd_model, csd_info = load_csd(device)
        log(f"CSD loaded: {csd_info['file']} sha256 {csd_info['file_sha256']}")
    if want_gram:
        vgg_features, vgg_info = load_vgg(device)
        if smoke:
            fit_images = indices
            fit_info = {"rule": (f"SMOKE stand-in: PCA fitted on the {SMOKE_IMAGES} smoke images "
                                 f"(images 0..{SMOKE_IMAGES - 1})"), "fit_images": int(len(fit_images))}
        else:
            fit_images, fit_info = pca_fit_images(data, row_to_image)
        pcas, check, layer_info, fit_timing = fit_gram_pca(vgg_features, image_paths, fit_images, device,
                                                           batch_size, workers)
        timing.update(fit_timing)
        projector = GramProjector(pcas, device)
        rel = gpu_vs_sklearn(projector, pcas, check, device)
        assert rel < 1e-4, f"GPU projection differs from sklearn PCA.transform (max relative diff {rel})"
        log(f"GPU projection vs sklearn transform: max relative diff {rel:.2e}")
        gram_extra = {"fit": fit_info, "layers": layer_info, "gpu_vs_sklearn_max_rel_diff": rel,
                      "fit_images": fit_images, "check": check}
    del data

    csd_out, gram_out, pass2_seconds = extract_all(image_paths, indices, csd_model, vgg_features, projector, device,
                                                   batch_size, workers)
    timing["pass2_seconds"] = round(pass2_seconds, 3)
    timing["pass2_images_per_second"] = round(len(indices) / pass2_seconds, 2)
    writes = []

    if want_csd:
        stats = summarize("CSD style", csd_out)
        spot = spot_check(image_paths, indices, csd_out, lambda x: csd_embed(csd_model, x.to(device)), True)
        assert spot > 0.999, f"CSD spot re-encode min cosine {spot}"
        log(f"CSD spot re-encode min cosine {spot:.6f}")
        meta = dict(common, kind="CSD style embedding", model=csd_info, preprocessing=CSD_PREPROCESSING,
                    output={"embeddings.npy": stats, "spot_reencode_min_cosine": spot},
                    batch_size=batch_size, workers=workers, timing=dict(timing, pass2_includes_gram=want_gram))
        writes.append((csd_dir, csd_out, meta, None))

    if want_gram:
        stats = summarize("VGG-Gram", gram_out)
        block_norms = np.linalg.norm(gram_out.reshape(len(gram_out), len(RELU_LAYERS), PCA_DIM), axis=2)
        assert np.allclose(block_norms, 1 / np.sqrt(len(RELU_LAYERS)), atol=1e-4), "Gram blocks not equal-norm"
        fit_images, check = gram_extra.pop("fit_images"), gram_extra.pop("check")
        expected = projector([torch.as_tensor(c, device=device) for c in check]).cpu().numpy()
        rows = np.searchsorted(indices, fit_images[:CHECK_ROWS])
        end_to_end = float((expected * gram_out[rows]).sum(axis=1).min())
        assert end_to_end > 0.999, f"pass 1 vs pass 2 Gram descriptors disagree (min cosine {end_to_end})"
        spot = spot_check(image_paths, indices, gram_out,
                          lambda x: projector(gram_upper(vgg_features, x.to(device))), False)
        assert spot > 0.999, f"Gram spot re-encode min cosine {spot}"
        log(f"Gram: pass1-vs-pass2 min cosine {end_to_end:.6f}; spot re-encode min cosine {spot:.6f}")
        meta = dict(common, kind="VGG-19 Gram descriptor (per-layer PCA 128, block L2, concat 640, L2)",
                    model=vgg_info, preprocessing=VGG_PREPROCESSING,
                    gram={"layers": {name: {"features_index": idx, "channels": c} for name, (idx, c)
                                     in RELU_LAYERS.items()},
                          "normalisation": "G = F F^T / (H W) per layer, F = (C, H*W) post-ReLU activations, float32",
                          "vector": "upper triangle with diagonal, torch.triu_indices(C, C) row-major order",
                          "pca": {"sklearn": "PCA(n_components=128, svd_solver='randomized', random_state=0, "
                                             "copy=False), other parameters default",
                                  "projection": "(g - mean_) @ components_.T on the GPU in float32 (whiten=False)",
                                  **gram_extra},
                          "final": "each 128-d block L2-normalised; blocks concatenated in layer order "
                                   "relu1_1..relu5_1 (640); L2-normalised (each block then has norm 1/sqrt(5))",
                          "fit_image_indices": fit_images.tolist(), "pca_file": "gram_pca.npz"},
                    output={"embeddings.npy": stats, "pass1_vs_pass2_min_cosine": end_to_end,
                            "spot_reencode_min_cosine": spot},
                    batch_size=batch_size, workers=workers, timing=dict(timing, pass2_includes_csd=want_csd))
        pca_arrays = {f"{name}_{field}": getattr(p, field) for name, p in zip(RELU_LAYERS, pcas)
                      for field in ("mean_", "components_", "explained_variance_ratio_")}
        writes.append((gram_dir, gram_out, meta, dict(pca_arrays, fit_image_indices=fit_images)))

    # Every check above has passed; only now is anything written (temporary file, then rename).
    for out_dir, array, meta, pca_arrays in writes:
        if pca_arrays is not None:
            np.savez(out_dir / "gram_pca.tmp.npz", **pca_arrays)
            (out_dir / "gram_pca.tmp.npz").replace(out_dir / "gram_pca.npz")
        save_npy(out_dir / "embeddings.npy", array)
        if not smoke:
            save_npy(out_dir / "row_to_image.npy", row_to_image)
            save_npy(out_dir / "row_to_image_annotation_order.npy", row_to_image_ann)
        write_json(out_dir / "meta.json", meta)
        log(f"wrote {out_dir}")
    log(f"{mode.upper()} PASS ({which}); total wall-clock {perf_counter() - started:.1f} s; timing {timing}")


def download() -> None:
    """Fetch both weight files into /data/SSD2 and verify them (needs network)."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(CSD_REPO_ID, CSD_FILENAME, revision=CSD_REVISION, cache_dir=str(HF_CACHE)))
    digest = sha256_file(path)
    if digest != CSD_SHA256 or path.stat().st_size != CSD_BYTES:
        raise AssertionError(f"CSD download {path}: sha256 {digest}, {path.stat().st_size} bytes; expected "
                             f"{CSD_SHA256}, {CSD_BYTES}")
    log(f"CSD checkpoint OK: {path} sha256 {digest}")
    torch.hub.set_dir(str(VGG_HUB_DIR))
    torch.hub.load_state_dict_from_url(VGG_WEIGHTS.url, progress=True, check_hash=True)
    log(f"VGG-19 weights OK: {VGG_FILE} sha256 {sha256_file(VGG_FILE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["download", "smoke", "full"])
    parser.add_argument("--which", choices=["both", "csd", "gram"], default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.workers <= 8:
        parser.error("batch size must be positive and workers in 0..8 (shared-machine rule)")
    if args.mode == "download":
        assert_plan()
        download()
    else:
        run(args.mode, args.which, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
