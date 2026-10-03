"""WikiArt genre labels (ArtGAN `WikiArt Dataset/Genre/genre_{train,val}.csv`) joined to ArtELingo paintings.

Downloaded 2026-10-02 into /data/SSD/wikiart_genre/. The class names come from ArtGAN's `Genre/genre_class`
file (no .txt extension; see docs/reports/auto/v2/2026-10-28_citation_check.md, E1).
"""

import csv
from pathlib import Path

import numpy as np

GENRE_DIR = Path("/data/SSD/wikiart_genre")
GENRE_CSVS = (GENRE_DIR / "genre_train.csv", GENRE_DIR / "genre_val.csv")
GENRE_NAMES = ["abstract_painting", "cityscape", "genre_painting", "illustration", "landscape", "nude_painting",
               "portrait", "religious_painting", "sketch_and_study", "still_life"]


def load_wikiart_genre(paintings: np.ndarray, csv_paths=GENRE_CSVS) -> np.ndarray:
    """One genre id per entry of ``paintings`` (file stems); -1 when unknown or when the CSVs disagree."""
    ids: dict[str, int] = {}
    for path in csv_paths:
        with Path(path).open() as handle:
            for rel_path, label in csv.reader(handle):
                stem = Path(rel_path).stem
                value = int(label)
                if stem in ids and ids[stem] != value:
                    ids[stem] = -1
                else:
                    ids.setdefault(stem, value)
    return np.asarray([ids.get(str(p), -1) for p in paintings], dtype=np.int64)
