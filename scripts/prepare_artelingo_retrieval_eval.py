"""Group flat ArtELingo val/test annotations for 1-image, 5-caption retrieval.

Usage: python scripts/prepare_artelingo_retrieval_eval.py
The flat training annotations are intentionally left unchanged.
"""

import json
import tempfile
from pathlib import Path


DATA_DIR = Path("/data/PDD/artelingo")
CAPTIONS_PER_IMAGE = 5


def build_retrieval_rows(rows):
    """Return ordered, eligible painting rows and the number of short paintings."""
    by_painting = {}
    for row in rows:
        by_painting.setdefault(row["painting"], []).append(row)

    output = []
    dropped = 0
    seen_images = set()
    for painting, painting_rows in by_painting.items():
        first = painting_rows[0]
        for row in painting_rows:
            if row["image"] != first["image"] or row["art_style"] != first["art_style"]:
                raise ValueError(f"Conflicting image or art_style for painting {painting!r}")
            if not isinstance(row["caption"], str):
                raise ValueError(f"Non-string caption for painting {painting!r}")

        if len(painting_rows) < CAPTIONS_PER_IMAGE:
            dropped += 1
            continue
        if first["image"] in seen_images:
            raise ValueError(f"Image shared by different paintings: {first['image']!r}")
        seen_images.add(first["image"])
        output.append({
            "image": first["image"],
            "caption": [row["caption"] for row in painting_rows[:CAPTIONS_PER_IMAGE]],
            "image_id": painting,
            "art_style": first["art_style"],
            "painting": painting,
        })
    return output, dropped


def verify_retrieval_file(path):
    """Read back a split and verify the retrieval shape and unique images/IDs."""
    with open(path, encoding="utf-8") as file:
        rows = json.load(file)
    images = set()
    image_ids = set()
    for row in rows:
        captions = row["caption"]
        if not isinstance(captions, list) or len(captions) != CAPTIONS_PER_IMAGE:
            raise ValueError(f"Expected {CAPTIONS_PER_IMAGE} captions for {row['image_id']!r}")
        if not all(isinstance(caption, str) for caption in captions):
            raise ValueError(f"Non-string caption for {row['image_id']!r}")
        if row["image"] in images:
            raise ValueError(f"Duplicate image: {row['image']!r}")
        if row["image_id"] in image_ids:
            raise ValueError(f"Duplicate image_id: {row['image_id']!r}")
        images.add(row["image"])
        image_ids.add(row["image_id"])
    return len(rows)


def main():
    prepared = []
    for split in ("val", "test"):
        source = DATA_DIR / f"artelingo_{split}.json"
        destination = DATA_DIR / f"artelingo_{split}_retrieval.json"
        with open(source, encoding="utf-8") as file:
            input_rows = json.load(file)
        output_rows, dropped = build_retrieval_rows(input_rows)
        prepared.append((split, source, destination, output_rows, dropped))

    staged = []
    try:
        for split, source, destination, output_rows, dropped in prepared:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=DATA_DIR,
                prefix=f".artelingo_{split}_retrieval.", suffix=".tmp", delete=False,
            ) as file:
                temporary_path = Path(file.name)
                staged.append((split, destination, temporary_path, dropped))
                json.dump(output_rows, file, ensure_ascii=False)
            temporary_path.chmod(source.stat().st_mode & 0o777)
            verify_retrieval_file(temporary_path)

        for split, destination, temporary_path, dropped in staged:
            temporary_path.replace(destination)
            count = verify_retrieval_file(destination)
            print(f"{split}: paintings kept={count}; paintings dropped={dropped} "
                  f"(insufficient captions, <{CAPTIONS_PER_IMAGE}); output rows={count}; "
                  f"all caption lists have exactly {CAPTIONS_PER_IMAGE} strings: yes")
            print(f"  wrote {destination}")
    finally:
        for _, _, temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
