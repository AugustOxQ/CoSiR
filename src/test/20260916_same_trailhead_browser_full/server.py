"""FastAPI server for the full, cached Same Trailhead browser."""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE, CACHE = Path(__file__).parent, Path(__file__).parent / "cache"
ANNOTATION_PATH = Path("/data/PDD/redcaps/redcaps_plus/redcaps_150k.json")
IMAGE_ROOT = Path("/data/PDD")
BUCKETS = ("unconnected", "txt_only", "img_only", "both")
ARRAY_NAMES = ("hub", "c", "d", "b", "edge_type_code", "hub_deg_txt_only", "hub_deg_img_only", "cd_img_cosine_distance", "sample_ids")
ARRAYS: dict[str, np.ndarray] = {}
ANNOTATIONS: list[dict] = []
BUCKET_ROWS: dict[str, np.ndarray] = {}
STATS: dict = {}
app = FastAPI(title="Same Trailhead — Full Browser")


def _subreddit(annotation: dict) -> str:
    parts = annotation["image"].split("/")
    if len(parts) < 3:
        raise ValueError(f"unexpected RedCaps image path: {annotation['image']}")
    return parts[2]


def _annotations_by_feature_position(annotations: list[dict], sample_ids: np.ndarray) -> list[dict]:
    if np.any(sample_ids < 0) or np.any(sample_ids >= len(annotations)):
        raise ValueError("feature-store sample IDs do not index the annotation list")
    return [dict(annotations[int(sample_id)], sample_id=int(sample_id)) for sample_id in sample_ids]


def _global_row(bucket: str, index: int) -> int:
    if bucket not in BUCKET_ROWS:
        raise ValueError(f"unknown bucket {bucket!r}; choose one of {', '.join(BUCKETS)}")
    rows = BUCKET_ROWS[bucket]
    if index < 0 or index >= len(rows):
        raise ValueError(f"index must be within [0, {len(rows) - 1}] for bucket {bucket!r}")
    return int(rows[index])


def _validate_cache(arrays: dict[str, np.ndarray], metadata: dict) -> None:
    row_count = int(metadata["row_count"])
    if any(len(arrays[name]) != row_count for name in ARRAY_NAMES if name != "sample_ids"):
        raise ValueError("cache arrays have inconsistent row counts")
    codes = arrays["edge_type_code"]
    if np.any((codes < 0) | (codes >= len(BUCKETS))):
        raise ValueError("cache contains unknown edge-type codes")
    expected = {entry["name"]: (int(entry["code"]), int(entry["count"])) for entry in metadata["buckets"]}
    if set(expected) != set(BUCKETS) or any(expected[name][0] != code for code, name in enumerate(BUCKETS)):
        raise ValueError("cache metadata has invalid bucket mapping")
    actual = {name: int((codes == code).sum()) for code, name in enumerate(BUCKETS)}
    if sum(actual.values()) != row_count or any(actual[name] != expected[name][1] for name in BUCKETS):
        raise ValueError("cache metadata bucket counts do not cover every row")


def _data_uri(image_path: Path, image_size: int = 160) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def _item(position: int) -> dict:
    annotation = ANNOTATIONS[position]
    image_path = IMAGE_ROOT / annotation["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing source image: {image_path}")
    return {"sample_id": annotation["sample_id"], "caption": annotation["caption"], "image_id": annotation["image_id"],
            "subreddit": _subreddit(annotation), "data_uri": _data_uri(image_path)}


def _example(bucket: str, index: int) -> dict:
    row = _global_row(bucket, index)
    return {"bucket": bucket, "index": index, "total": int(len(BUCKET_ROWS[bucket])), "global_row": row,
            "hub_deg_txt_only": int(ARRAYS["hub_deg_txt_only"][row]), "hub_deg_img_only": int(ARRAYS["hub_deg_img_only"][row]),
            "cd_img_cosine_distance": float(ARRAYS["cd_img_cosine_distance"][row]),
            "A": _item(int(ARRAYS["hub"][row])), "B": _item(int(ARRAYS["b"][row])),
            "C": _item(int(ARRAYS["c"][row])), "D": _item(int(ARRAYS["d"][row]))}


@app.on_event("startup")
async def startup() -> None:
    global ARRAYS, ANNOTATIONS, BUCKET_ROWS, STATS
    if not (CACHE / "metadata.json").is_file():
        raise RuntimeError(f"Missing cache at {CACHE}; run build_full_index.py first")
    ARRAYS = {name: np.load(CACHE / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}
    metadata = json.loads((CACHE / "metadata.json").read_text())
    _validate_cache(ARRAYS, metadata)
    with ANNOTATION_PATH.open() as source:
        ANNOTATIONS = _annotations_by_feature_position(json.load(source), ARRAYS["sample_ids"])
    BUCKET_ROWS = {name: np.flatnonzero(ARRAYS["edge_type_code"] == code) for code, name in enumerate(BUCKETS)}
    old_stats = json.loads((HERE.parent / "20260901_same_trailhead_browser" / "abcd_examples.json").read_text())["edge_type_stats"]
    STATS = {name: {**old_stats[name], "count": int(len(BUCKET_ROWS[name]))} for name in BUCKETS}


@app.get("/api/buckets")
def buckets() -> dict:
    return {"buckets": [{"name": name, **STATS[name], "pull_context": "Measured in a separate trained-embedding experiment; shown for context and not recomputed per pair below."} for name in BUCKETS]}


@app.get("/api/example")
def example(bucket: str = Query(...), index: int = Query(...)) -> dict:
    try:
        return _example(bucket, index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/random")
def random_example(bucket: str = Query(...)) -> dict:
    try:
        if bucket not in BUCKET_ROWS:
            raise ValueError(f"unknown bucket {bucket!r}; choose one of {', '.join(BUCKETS)}")
        total = len(BUCKET_ROWS[bucket])
        return _example(bucket, int(np.random.default_rng().integers(total)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def root() -> FileResponse:
    return FileResponse(HERE / "browser_full.html")
