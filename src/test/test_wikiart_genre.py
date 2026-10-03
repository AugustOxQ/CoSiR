import numpy as np

from src.data.wikiart_genre import GENRE_NAMES, load_wikiart_genre


def test_genre_join_by_stem_and_conflicts(tmp_path):
    a = tmp_path / "genre_train.csv"
    b = tmp_path / "genre_val.csv"
    a.write_text("Impressionism/monet_water-lilies.jpg,4\nBaroque/rubens_x.jpg,7\nCubism/conflict.jpg,0\n")
    b.write_text("Cubism/conflict.jpg,6\nRealism/only-in-val.jpg,9\n")
    paintings = np.array(["monet_water-lilies", "rubens_x", "conflict", "only-in-val", "unknown"], dtype=object)
    got = load_wikiart_genre(paintings, csv_paths=(a, b))
    assert got.tolist() == [4, 7, -1, 9, -1]                 # conflict and unknown are missing


def test_genre_names_match_artgan_class_file():
    assert GENRE_NAMES[0] == "abstract_painting" and GENRE_NAMES[5] == "nude_painting"
    assert GENRE_NAMES[9] == "still_life" and len(GENRE_NAMES) == 10
