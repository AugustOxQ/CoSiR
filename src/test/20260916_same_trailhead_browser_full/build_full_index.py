"""Build the full Same Trailhead A/B/C/D index once, then cache compact arrays.

Run from the repository root with the CoSiR environment.  This reconstructs the
same K=30, alpha=0.5 graph semantics as the 2026-09-01 illustrative browser;
it does not train a model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges, pairwise_cosine_distances
from src.conditional_buddy.compute_buddies import _l2_normalize, build_buddy_graphs
from src.utils import FeatureManager


BUCKETS = ("unconnected", "txt_only", "img_only", "both")
EDGE_TYPE_CODES = {name: code for code, name in enumerate(BUCKETS)}
REFERENCE_COUNTS = {"unconnected": 4_284_528, "txt_only": 4_423_995, "img_only": 70_544, "both": 161_907}


def _load_features(storage_dir: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    data = FeatureManager(storage_dir).load_all_to_ram(["img_features", "txt_features"])
    return (data["img_features"].numpy().astype(np.float32), data["txt_features"].numpy().astype(np.float32),
            [int(sample_id) for sample_id in data["sample_ids"].tolist()])


def _edge_type_codes(typed: dict, n_nodes: int, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Classify C/D pairs using the existing sorted typed-edge representation."""
    keys = typed["keys"]
    pair_keys = np.minimum(c, d).astype(np.int64) * n_nodes + np.maximum(c, d).astype(np.int64)
    positions = np.searchsorted(keys, pair_keys)
    safe_positions = positions.clip(max=max(len(keys) - 1, 0))
    present = (positions < len(keys)) & (keys[safe_positions] == pair_keys)
    repair = present & typed["repair"][safe_positions]
    if repair.any():
        raise RuntimeError(f"C/D enumeration contains {int(repair.sum()):,} repair edges; expected only four real-edge buckets")
    codes = np.zeros(len(pair_keys), dtype=np.uint8)
    for name in ("txt_only", "img_only", "both"):
        code = EDGE_TYPE_CODES[name]
        codes[present & typed[name][safe_positions]] = code
    return codes


def _neighbors(typed: dict, n_nodes: int, edge_name: str) -> dict[int, list[int]]:
    keys = typed["keys"]
    left, right = (keys // n_nodes).astype(np.int64), (keys % n_nodes).astype(np.int64)
    result: dict[int, list[int]] = {}
    for a, b in zip(left[typed[edge_name]].tolist(), right[typed[edge_name]].tolist()):
        result.setdefault(a, []).append(b)
        result.setdefault(b, []).append(a)
    return result


def _write_array(cache_dir: Path, name: str, array: np.ndarray) -> int:
    path = cache_dir / f"{name}.npy"
    np.save(path, array)
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", default="/data/SSD2/pre_extract/redcaps_150k/features")
    parser.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_features, txt_features, sample_ids = _load_features(args.storage_dir)
    img_normalized, txt_normalized = _l2_normalize(img_features), _l2_normalize(txt_features)
    a_img, a_txt, union = build_buddy_graphs(img_normalized, txt_normalized, K=args.K, alpha=args.alpha, device=args.device)
    n_nodes = len(sample_ids)
    typed = classify_edges(a_img, a_txt, union, n_nodes)
    stats = bridge_node_stats(typed, n_nodes)
    txt_neighbors, img_neighbors = _neighbors(typed, n_nodes, "txt_only"), _neighbors(typed, n_nodes, "img_only")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ("hub", "c", "d", "b", "edge_type_code", "hub_deg_txt_only", "hub_deg_img_only")}
    for hub in np.flatnonzero((stats["deg_txt_only"] >= 2) & (stats["deg_img_only"] > 0)):
        neighbors = np.asarray(txt_neighbors[int(hub)], dtype=np.int32)
        left, right = np.triu_indices(len(neighbors), k=1)
        c, d = neighbors[left], neighbors[right]
        count = len(c)
        b = int(np.random.default_rng(args.seed + int(hub)).choice(img_neighbors[int(hub)]))
        chunks["hub"].append(np.full(count, hub, dtype=np.int32))
        chunks["c"].append(c)
        chunks["d"].append(d)
        chunks["b"].append(np.full(count, b, dtype=np.int32))
        chunks["edge_type_code"].append(_edge_type_codes(typed, n_nodes, c, d))
        chunks["hub_deg_txt_only"].append(np.full(count, stats["deg_txt_only"][hub], dtype=np.int32))
        chunks["hub_deg_img_only"].append(np.full(count, stats["deg_img_only"][hub], dtype=np.int32))
    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}
    arrays["cd_img_cosine_distance"] = pairwise_cosine_distances(img_normalized, arrays["c"], arrays["d"]).astype(np.float32)
    counts = {name: int((arrays["edge_type_code"] == code).sum()) for name, code in EDGE_TYPE_CODES.items()}
    matches_reference = counts == REFERENCE_COUNTS
    if not matches_reference:
        print("WARNING: FULL-POPULATION BUCKET MISMATCH")
        print(f"  expected: {REFERENCE_COUNTS}")
        print(f"  rebuilt:  {counts}")

    bytes_written = {name: _write_array(cache_dir, name, value) for name, value in arrays.items()}
    bytes_written["sample_ids"] = _write_array(cache_dir, "sample_ids", np.asarray(sample_ids, dtype=np.int32))
    metadata = {"buckets": [{"name": name, "code": code, "count": counts[name]} for name, code in EDGE_TYPE_CODES.items()],
                "K": args.K, "alpha": args.alpha, "seed": args.seed, "row_count": int(len(arrays["hub"])),
                "reference_counts": REFERENCE_COUNTS, "matches_reference_counts": matches_reference,
                "array_bytes": bytes_written}
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    elapsed = time.perf_counter() - started
    print(f"Built {len(arrays['hub']):,} rows in {elapsed:.2f}s; cache bytes={sum(bytes_written.values()):,}")
    print("Per-bucket rows:", counts)


if __name__ == "__main__":
    main()
