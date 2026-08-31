"""Rebuild the illustrative Same Trailhead A/B/C/D browser payload.

This is deliberately separate from Experiment 14's analysis pipeline.  It
reconstructs the same K=30, alpha=0.5 buddy graph, preserves C/D's graph bucket
as the independent variable, and applies an *additive* CLIP-image cosine-distance
eligibility rule only to genuinely-unconnected C/D candidates.

Run from the repository root with the project environment, for example:

  /root/miniconda3/envs/CoSiR/bin/python \
    src/test/20260901_same_trailhead_browser/generate_abcd_examples.py \
    --storage-dir /data/SSD2/pre_extract/redcaps_150k/features \
    --annotation-path /data/PDD/redcaps/redcaps_plus/redcaps_150k.json \
    --image-root /data/PDD --device cuda --min-cd-image-distance 0.6
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.conditional_buddy.buddy_graph import (
    bridge_node_stats,
    classify_edges,
    minimum_cosine_distance_mask,
    pairwise_cosine_distances,
)
from src.conditional_buddy.compute_buddies import _l2_normalize, build_buddy_graphs
from src.utils import FeatureManager


EDGE_TYPES = ("unconnected", "txt_only", "img_only", "both")


def _load_features(storage_dir: str):
    feature_manager = FeatureManager(storage_dir)
    data = feature_manager.load_all_to_ram(["img_features", "txt_features"])
    return (
        data["img_features"].numpy().astype(np.float32),
        data["txt_features"].numpy().astype(np.float32),
        [int(sample_id) for sample_id in data["sample_ids"].tolist()],
    )


def _edge_type_labels(typed: dict, N: int, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Return the five-way direct C/D relationship without changing graph labels."""
    keys = typed["keys"]
    pair_keys = np.minimum(c, d).astype(np.int64) * N + np.maximum(c, d).astype(np.int64)
    labels = np.full(len(pair_keys), "unconnected", dtype="<U12")
    positions = np.searchsorted(keys, pair_keys)
    present = (positions < len(keys)) & (keys[np.minimum(positions, len(keys) - 1)] == pair_keys)
    for edge_type in ("txt_only", "img_only", "both", "repair"):
        matches = present & typed[edge_type][positions.clip(max=len(keys) - 1)]
        labels[matches] = edge_type
    return labels


def _data_uri(image_path: Path, image_size: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def _browser_item(annotation: dict, image_root: Path, image_size: int) -> dict:
    image_path = image_root / annotation["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing source image: {image_path}")
    return {
        "sample_id": annotation["sample_id"],
        "caption": annotation["caption"],
        "image_id": annotation["image_id"],
        "data_uri": _data_uri(image_path, image_size),
    }


def _annotations_by_feature_position(annotations: list[dict], sample_ids: list[int]) -> list[dict]:
    """Resolve shuffled feature rows to their original RedCaps annotations."""
    if any(sample_id < 0 or sample_id >= len(annotations) for sample_id in sample_ids):
        raise ValueError("feature-store sample IDs do not index the annotation list")
    return [dict(annotations[sample_id], sample_id=sample_id) for sample_id in sample_ids]


def _choose_indices(
    labels: np.ndarray,
    hub: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    cd_distance: np.ndarray,
    eligible: np.ndarray,
    examples_per_bucket: int,
    rng: np.random.Generator,
) -> list[int]:
    """Choose a balanced, distinct-hub browser subset from eligible candidates."""
    chosen, used_hubs, used_cd_pairs = [], set(), set()
    for edge_type in EDGE_TYPES:
        indices = np.flatnonzero((labels == edge_type) & eligible)
        if edge_type == "unconnected":
            indices = indices[np.argsort(cd_distance[indices])[::-1]]
        else:
            indices = rng.permutation(indices)

        selected = []
        for index in indices:
            cd_pair = (min(int(c[index]), int(d[index])), max(int(c[index]), int(d[index])))
            if int(hub[index]) not in used_hubs and cd_pair not in used_cd_pairs:
                selected.append(int(index))
                used_hubs.add(int(hub[index]))
                used_cd_pairs.add(cd_pair)
            if len(selected) == examples_per_bucket:
                break
        if len(selected) != examples_per_bucket:
            raise RuntimeError(
                f"Only {len(selected)} distinct hubs available for {edge_type}; "
                f"need {examples_per_bucket}. Lower the threshold or reduce the requested examples."
            )
        chosen.extend(selected)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--stats-source", default="src/test/20260901_same_trailhead_browser/abcd_examples.json")
    parser.add_argument("--output-json", default="src/test/20260901_same_trailhead_browser/abcd_examples.json")
    parser.add_argument("--output-js", default="src/test/20260901_same_trailhead_browser/data.js")
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-cd-image-distance", type=float, default=0.6)
    parser.add_argument("--examples-per-bucket", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    img_features, txt_features, sample_ids = _load_features(args.storage_dir)
    img_normalized = _l2_normalize(img_features)
    txt_normalized = _l2_normalize(txt_features)
    A_img, A_txt, union = build_buddy_graphs(
        img_normalized, txt_normalized, K=args.K, alpha=args.alpha, device=args.device,
    )
    N = len(sample_ids)
    typed = classify_edges(A_img, A_txt, union, N)
    stats = bridge_node_stats(typed, N)

    keys = typed["keys"]
    row = (keys // N).astype(np.int64)
    col = (keys % N).astype(np.int64)
    txt_row, txt_col = row[typed["txt_only"]], col[typed["txt_only"]]
    txt_neighbors: dict[int, list[int]] = {}
    for left, right in zip(txt_row.tolist(), txt_col.tolist()):
        txt_neighbors.setdefault(left, []).append(right)
        txt_neighbors.setdefault(right, []).append(left)

    hubs, cs, ds = [], [], []
    for hub in np.flatnonzero((stats["deg_txt_only"] >= 2) & (stats["deg_img_only"] > 0)):
        neighbors = txt_neighbors.get(int(hub), [])
        for c_position in range(len(neighbors)):
            for d_position in range(c_position + 1, len(neighbors)):
                hubs.append(int(hub))
                cs.append(neighbors[c_position])
                ds.append(neighbors[d_position])
    hub = np.asarray(hubs, dtype=np.int64)
    c = np.asarray(cs, dtype=np.int64)
    d = np.asarray(ds, dtype=np.int64)
    labels = _edge_type_labels(typed, N, c, d)
    cd_distance = pairwise_cosine_distances(img_normalized, c, d)

    eligible = np.ones(len(hub), dtype=bool)
    unconnected = labels == "unconnected"
    eligible[unconnected] = minimum_cosine_distance_mask(
        img_normalized, c[unconnected], d[unconnected], args.min_cd_image_distance,
    )
    selected = _choose_indices(
        labels, hub, c, d, cd_distance, eligible, args.examples_per_bucket,
        np.random.default_rng(args.seed),
    )

    with open(args.annotation_path) as annotation_file:
        annotations = json.load(annotation_file)
    annotations = _annotations_by_feature_position(annotations, sample_ids)
    image_root = Path(args.image_root)
    img_only_neighbors: dict[int, list[int]] = {}
    for left, right in zip(row[typed["img_only"]].tolist(), col[typed["img_only"]].tolist()):
        img_only_neighbors.setdefault(left, []).append(right)
        img_only_neighbors.setdefault(right, []).append(left)

    examples = []
    for output_index, candidate_index in enumerate(selected):
        a = int(hub[candidate_index])
        b = int(np.random.default_rng(args.seed + a).choice(img_only_neighbors[a]))
        edge_type = str(labels[candidate_index])
        examples.append({
            "index": output_index,
            "edge_type": edge_type,
            "hub_deg_txt_only": int(stats["deg_txt_only"][a]),
            "hub_deg_img_only": int(stats["deg_img_only"][a]),
            "selection": {
                "cd_img_cosine_distance": float(cd_distance[candidate_index]),
                "min_cd_img_cosine_distance": args.min_cd_image_distance if edge_type == "unconnected" else None,
                "cd_is_union_unconnected": edge_type == "unconnected",
            },
            "A": _browser_item(annotations[a], image_root, args.image_size),
            "B": _browser_item(annotations[b], image_root, args.image_size),
            "C": _browser_item(annotations[int(c[candidate_index])], image_root, args.image_size),
            "D": _browser_item(annotations[int(d[candidate_index])], image_root, args.image_size),
        })

    with open(args.stats_source) as stats_file:
        edge_type_stats = json.load(stats_file)["edge_type_stats"]
    payload = {
        "selection_metadata": {
            "K": args.K,
            "alpha": args.alpha,
            "seed": args.seed,
            "unconnected_cd_img_min_distance": args.min_cd_image_distance,
            "selection_rule": "Graph bucket first; only union-unconnected C/D candidates require the additive image-distance filter.",
        },
        "examples": examples,
        "edge_type_stats": edge_type_stats,
    }
    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    Path(args.output_js).write_text("window.ABCD_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {output_json}")
    print("Unconnected C/D image distances:", [round(example["selection"]["cd_img_cosine_distance"], 4) for example in examples if example["edge_type"] == "unconnected"])


if __name__ == "__main__":
    main()
