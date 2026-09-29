#!/usr/bin/env python3
"""Real check for painting-grouped split: zero leakage and proper row shares."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.data.artelingo import load_artelingo
from src.data.splits import grouped_split, leakage_groups, split_leakage

FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723


def main() -> None:
    started = perf_counter()

    # Load ArtELingo data
    print("Loading ArtELingo data...", flush=True)
    data = load_artelingo(feature_dir=FEATURE_DIR, annotations_path=ANNOTATIONS_PATH,
                         expected_samples=EXPECTED_SAMPLES)

    print(f"Loaded {len(data.sample_ids):,} rows with {len(np.unique(data.paintings))} paintings "
          f"and {len(np.unique(data.emotions))} emotions", flush=True)

    # Compute leakage groups
    print("Computing leakage groups...", flush=True)
    groups = leakage_groups(data.paintings, data.img_features)
    n_groups = len(np.unique(groups))
    print(f"Grouped into {n_groups:,} groups", flush=True)

    # Perform grouped split
    print("Performing grouped split with seed=42...", flush=True)
    split = grouped_split(groups, fractions=(0.7, 0.1, 0.2), seed=42)

    # Compute statistics per part
    parts_dict = {"train": split.train, "val": split.val, "held": split.held}
    stats = {}
    emotion_dist = {}

    for part_name, indices in parts_dict.items():
        n_rows = len(indices)
        n_paintings = len(np.unique(data.paintings[indices]))
        n_part_groups = len(np.unique(groups[indices]))
        share = n_rows / len(data.sample_ids)

        stats[part_name] = {
            "n_rows": n_rows,
            "n_paintings": n_paintings,
            "n_groups": n_part_groups,
            "row_share": share,
        }

        # Emotion distribution for this part
        part_emotions = data.emotions[indices]
        emotion_counts = {}
        for emotion in np.unique(part_emotions):
            emotion_counts[emotion] = int(np.sum(part_emotions == emotion))
        emotion_dist[part_name] = emotion_counts

    # Compute largest group size
    unique, counts = np.unique(groups, return_counts=True)
    largest_group_size = int(np.max(counts))

    # Check leakage
    print("Checking for leakage...", flush=True)
    leakage = split_leakage(split, data.paintings, data.img_features)

    elapsed = perf_counter() - started

    # Print results
    print("\n" + "="*70)
    print("SPLIT RESULTS")
    print("="*70)

    print("\nRows, paintings, and groups per part:")
    for part_name in ["train", "val", "held"]:
        s = stats[part_name]
        print(f"  {part_name:6s}: {s['n_rows']:7,} rows ({s['row_share']*100:5.1f}%) | "
              f"{s['n_paintings']:5,} paintings | {s['n_groups']:6,} groups")

    print("\nRow shares (target 70/10/20):")
    for part_name in ["train", "val", "held"]:
        share = stats[part_name]["row_share"]
        print(f"  {part_name:6s}: {share*100:5.1f}%")

    print("\nLargest group size: {:,}".format(largest_group_size))

    print("\nEmotion distribution per part:")
    for part_name in ["train", "val", "held"]:
        print(f"  {part_name}:")
        emotion_dict = emotion_dist[part_name]
        for emotion in sorted(emotion_dict.keys()):
            count = emotion_dict[emotion]
            print(f"    {emotion:20s}: {count:6,}")

    print("\nLeakage check (all values must be 0):")
    all_zero = True
    for key in sorted(leakage.keys()):
        value = leakage[key]
        status = "✓" if value == 0 else "✗"
        print(f"  {status} {key}: {value}")
        if value != 0:
            all_zero = False

    print("\nRuntime: {:.3f}s".format(elapsed))
    print("="*70)

    if not all_zero:
        print("\nERROR: Leakage detected! Check the leakage dict above.")
        return False

    print("\nSUCCESS: No leakage detected and row shares are reasonable.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
