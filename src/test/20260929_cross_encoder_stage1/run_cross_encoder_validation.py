"""Full-train DINOv2 + e5 ablation using Block 1's unchanged pipeline."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
from src.model.graph import GraphConfig, build_content_graph
from src.train.stage1 import Stage1Config, train_stage1


FEATURE_DIR = Path(__file__).resolve().parent / "features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723


def pad_image_features(images: np.ndarray, text_dim: int) -> np.ndarray:
    """Embed DINOv2 in e5's width without changing image norms or distances."""
    if images.ndim != 2 or text_dim < images.shape[1]:
        raise ValueError("Image features must be 2-D and no wider than text features")
    padded = np.zeros((len(images), text_dim), dtype=np.float32)
    padded[:, : images.shape[1]] = images
    return padded


def raw_embeddings(images: np.ndarray, texts: np.ndarray) -> np.ndarray:
    """Mirror Block 1's raw baseline: normalize each view, then concatenate."""
    img_unit = images / np.linalg.norm(images, axis=1, keepdims=True)
    txt_unit = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    result = np.concatenate((img_unit, txt_unit), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Raw features contain non-finite values")
    return result


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use Task 1's positional alignment with the ArtELingo annotation list."""
    images = np.load(FEATURE_DIR / "dinov2_img.npy", mmap_mode="r")
    texts = np.load(FEATURE_DIR / "e5_txt.npy", mmap_mode="r")
    with ANNOTATIONS_PATH.open() as file:
        annotations = json.load(file)
    if (images.shape != (EXPECTED_SAMPLES, 384)
            or texts.shape != (EXPECTED_SAMPLES, 768)
            or len(annotations) != EXPECTED_SAMPLES
            or images.dtype != np.float32 or texts.dtype != np.float32):
        raise ValueError("Feature shapes/dtypes and annotation rows must match Task 1")
    emotions = np.asarray([row["emotion"] for row in annotations])
    return images, texts, emotions


def score_partition(name: str, embeddings: np.ndarray, emotions: np.ndarray) -> None:
    """Detect and summarize the same default Leiden communities as Block 1."""
    started = perf_counter()
    labels = detect_communities(embeddings)
    seconds = perf_counter() - started
    stats = community_stats(labels)
    ami = adjusted_mutual_info_score(emotions, labels)
    print(json.dumps({
        "variant": name,
        "emotion_ami": round(float(ami), 9),
        "num_communities": stats["num_communities"],
        "min_size": stats["min_size"],
        "median_size": float(np.median(stats["sizes"])),
        "max_size": stats["max_size"],
        "empty_count": stats["empty_count"],
        "sizes": stats["sizes"],
        "community_seconds": round(seconds, 3),
    }), flush=True)


def main() -> None:
    started = perf_counter()
    images, texts, emotions = load_inputs()
    print(f"Loaded {len(images):,} paired rows and {len(np.unique(emotions))} emotions", flush=True)

    graph_started = perf_counter()
    graph = build_content_graph(images, texts, GraphConfig())
    print(f"Graph edges={graph.nnz // 2:,}; seconds={perf_counter() - graph_started:.3f}",
          flush=True)

    # Stage 1 demands equal input widths; zero-padding is a fixed isometry.
    # The teacher graph above uses both untouched original feature arrays.
    padded_images = pad_image_features(images, texts.shape[1])
    train_started = perf_counter()
    _, trained = train_stage1(padded_images, texts, graph, Stage1Config())
    print(f"Trained embeddings={trained.shape}; seconds={perf_counter() - train_started:.3f}",
          flush=True)
    if not np.isfinite(trained).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    score_partition("trained_stage1", trained, emotions)
    del trained, graph, padded_images

    raw = raw_embeddings(images, texts)
    print(f"Raw embeddings={raw.shape}", flush=True)
    score_partition("raw_features", raw, emotions)
    print(f"Total seconds={perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
