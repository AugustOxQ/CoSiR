"""One-off epoch sensitivity check for content-only Stage 1 on ArtELingo train."""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
from src.model.graph import GraphConfig, build_content_graph
from src.train.stage1 import Stage1Config, train_stage1
from src.utils import FeatureManager


FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723
RAW_CLIP_AMI = 0.035781
EPOCH_COUNTS = (200, 2_000, 10_000)


def main() -> None:
    started = perf_counter()
    manager = FeatureManager(storage_dir=FEATURE_DIR)
    with ANNOTATIONS_PATH.open() as file:
        annotations = json.load(file)
    if not (len(annotations) == manager.total_samples == EXPECTED_SAMPLES):
        raise ValueError(
            "Annotation/feature count mismatch: "
            f"annotations={len(annotations)}, features={manager.total_samples}, "
            f"expected={EXPECTED_SAMPLES}"
        )

    features = manager.load_all_to_ram(["img_features", "txt_features"])
    sample_ids = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    img_features = features["img_features"].numpy()
    txt_features = features["txt_features"].numpy()
    if not (len(sample_ids) == len(img_features) == len(txt_features) == EXPECTED_SAMPLES):
        raise ValueError("Feature rows and sample IDs do not match metadata")
    if (sample_ids.min() < 0 or sample_ids.max() >= len(annotations)
            or len(np.unique(sample_ids)) != len(sample_ids)):
        raise ValueError("Sample IDs must uniquely index the annotation list")
    emotions = np.asarray([annotations[int(index)]["emotion"] for index in sample_ids])
    print(
        f"Loaded {len(sample_ids):,} samples; {len(np.unique(emotions))} emotion labels; "
        f"load_seconds={perf_counter() - started:.3f}",
        flush=True,
    )

    graph_started = perf_counter()
    graph = build_content_graph(img_features, txt_features, GraphConfig())
    print(
        f"Graph: nodes={graph.shape[0]:,}, edges={graph.nnz // 2:,}, "
        f"seconds={perf_counter() - graph_started:.3f}",
        flush=True,
    )

    results = []
    for epochs in EPOCH_COUNTS:
        training_started = perf_counter()
        loss_output = io.StringIO()
        with redirect_stdout(loss_output):
            _, embeddings = train_stage1(
                img_features, txt_features, graph, Stage1Config(epochs=epochs, seed=42)
            )
        losses = loss_output.getvalue().splitlines()
        if len(losses) != epochs:
            raise ValueError(f"Expected {epochs} loss lines, got {len(losses)}")
        first_loss = float(losses[0].split("loss=")[1])
        final_loss = float(losses[-1].split("loss=")[1])
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Stage 1 produced non-finite embeddings at {epochs} epochs")
        print(
            f"Training {epochs:,} epochs: first_loss={first_loss:.6f}, "
            f"final_loss={final_loss:.6f}, seconds={perf_counter() - training_started:.3f}",
            flush=True,
        )

        detection_started = perf_counter()
        labels = detect_communities(embeddings)
        stats = community_stats(labels)
        ami = adjusted_mutual_info_score(emotions, labels)
        results.append((epochs, final_loss, stats["num_communities"], ami))
        print(
            f"Communities {epochs:,} epochs: count={stats['num_communities']:,}, "
            f"min/median/max={stats['min_size']:,}/"
            f"{np.median(stats['sizes']):g}/{stats['max_size']:,}, "
            f"empty={stats['empty_count']}, emotion_ami={ami:.6f}, "
            f"gap={ami - RAW_CLIP_AMI:+.6f}, "
            f"seconds={perf_counter() - detection_started:.3f}",
            flush=True,
        )

    print("\nEpoch sensitivity (seed=42; raw-CLIP AMI=0.035781)")
    print("epochs | final training loss | communities | emotion AMI | gap over raw CLIP")
    for epochs, loss, communities, ami in results:
        print(f"{epochs:,} | {loss:.6f} | {communities:,} | {ami:.6f} | {ami - RAW_CLIP_AMI:+.6f}")
    print(f"Total seconds: {perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
