"""Convert the raw ArtELingo CSV releases into CoSiR-style annotation JSONs.

Source data (downloaded via gdown from the ArtELingo authors' Drive folder):
  /data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv
    - main trilingual (english/chinese/arabic) set, ~1.1M rows, one row per
      (painting, single caption, single emotion) annotation.
  /data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv
    - 28-language SemEval extension; carries an additional ground-truth
      `genre` label alongside `emotion`. Used only as a diagnostic set (buddy-
      graph community vs. genre/emotion ground-truth correlation), not for
      feature-extraction-at-scale training.

Images are NOT re-downloaded: both CSVs' image paths resolve directly against
the WikiArt dump already extracted at /data/PDD/wikiart_proj/wikiart/
(<art_style>/<painting>.jpg), so `train_image_path` in the dataset config
points there and `image` fields here are relative to it.

Split handling (per user decision, 2026-09-22):
  - English rows only (CLIP's text tower is English-tuned).
  - `rest` and blank splits are DROPPED, not folded into train: the official
    HuggingFace ArtELingo dataset card only lists train/test/val (920K/94.1K/
    46.9K, matching our train/test/val counts almost exactly) with no `rest`
    category, meaning the dataset's own release already excludes `rest` from
    all three official splits. We follow that precedent rather than
    second-guessing an unknown exclusion.
  - `val` is wired to CoSiR's `test_annotation_path` (used for periodic
    in-training eval snapshots); true `test` is written out but reserved for
    final reporting only, not referenced by the training config.

Usage:
    python scripts/prepare_artelingo_annotations.py
"""

import csv
import json
import os
from collections import Counter

MAIN_CSV = "/data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv"
GENRE_CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"
WIKIART_ROOT = "/data/PDD/wikiart_proj/wikiart"
OUT_DIR = "/data/PDD/artelingo"

DROPPED_SPLITS = {"rest", ""}


def _image_exists(rel_path: str) -> bool:
    return os.path.exists(os.path.join(WIKIART_ROOT, rel_path))


def build_main_splits():
    records = {"train": [], "val": [], "test": []}
    painting_counters = Counter()
    dropped_split = Counter()
    dropped_missing_image = 0

    with open(MAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"] != "english":
                continue
            split = row["split"]
            if split in DROPPED_SPLITS:
                dropped_split[split or "<blank>"] += 1
                continue
            if split not in records:
                dropped_split[split] += 1
                continue

            rel_image = f"{row['art_style']}/{row['painting']}.jpg"
            if not _image_exists(rel_image):
                dropped_missing_image += 1
                continue

            painting_counters[row["painting"]] += 1
            record = {
                "image": rel_image,
                "caption": row["utterance"],
                "image_id": f"{row['painting']}#{painting_counters[row['painting']]}",
                "emotion": row["emotion"],
                "art_style": row["art_style"],
                "painting": row["painting"],
            }
            records[split].append(record)

    print("=== main ArtELingo (English) ===")
    for split, rows in records.items():
        print(f"  {split}: {len(rows)} rows")
    print(f"  dropped (split filter): {dict(dropped_split)}")
    print(f"  dropped (missing image): {dropped_missing_image}")

    for split, rows in records.items():
        out_path = os.path.join(OUT_DIR, f"artelingo_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"  wrote {out_path} ({len(rows)} rows)")


def build_genre_emotion_diagnostic():
    records = []
    dropped_missing_image = 0

    with open(GENRE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"] != "english":
                continue
            rel_image = row["image_name"]
            if not _image_exists(rel_image):
                dropped_missing_image += 1
                continue
            records.append(
                {
                    "image": rel_image,
                    "caption": row["caption"],
                    "image_id": f"{row['painting']}#{row['image_id']}",
                    "emotion": row["emotion"],
                    "genre": row["genre"],
                    "art_style": row["art_style"],
                    "painting": row["painting"],
                    "split": row["split"],
                }
            )

    print("=== ArtELingo-28 genre+emotion diagnostic (English) ===")
    print(f"  rows: {len(records)}")
    print(f"  dropped (missing image): {dropped_missing_image}")

    out_path = os.path.join(OUT_DIR, "artelingo_genre_emotion_eng.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"  wrote {out_path} ({len(records)} rows)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_main_splits()
    print()
    build_genre_emotion_diagnostic()
