import numpy as np

from src.data.cub import dev_species, load_cub, zero_shot_split


def _fake_cub(root):
    (root / "CUB_200_2011" / "attributes").mkdir(parents=True)
    base = root / "CUB_200_2011"
    (base / "images.txt").write_text("1 001.A/a1.jpg\n2 001.A/a2.jpg\n3 002.B/b1.jpg\n")
    (base / "image_class_labels.txt").write_text("1 1\n2 1\n3 2\n")
    (base / "classes.txt").write_text("1 001.A\n2 002.B\n")
    (root / "attributes.txt").write_text("1 has_primary_color::red\n2 has_primary_color::blue\n3 has_size::small\n")
    rows = ["1 1 1 4 0.0", "1 2 0 4 0.0", "1 3 1 3 0.0",       # image 1: red only, small
            "2 1 1 4 0.0", "2 2 1 3 0.0", "2 3 0 3 0.0",       # image 2: two colours -> -1
            "3 1 0 4 0.0", "3 2 1 2 0.0", "3 3 1 4 0.0"]       # image 3: blue at certainty 2 -> -1
    (base / "attributes" / "image_attribute_labels.txt").write_text("\n".join(rows) + "\n")
    for cls, name in (("001.A", "a1"), ("001.A", "a2"), ("002.B", "b1")):
        d = root / "captions" / "extracted" / "text_c10" / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.txt").write_text("\n".join(f"caption {i}" for i in range(10)) + "\n")
    (root / "xlsa17").mkdir()
    (root / "xlsa17" / "trainvalclasses.txt").write_text("001.A\n")
    (root / "xlsa17" / "testclasses.txt").write_text("002.B\n")


def test_load_cub_attributes_and_split(tmp_path):
    _fake_cub(tmp_path)
    cub = load_cub(tmp_path)
    assert cub.attributes["has_primary_color"].tolist() == [0, -1, -1]      # red (code 0) only for image 1
    assert cub.attribute_values["has_primary_color"] == ["red", "blue"]      # file order, not alphabetical
    assert len(cub.captions[0]) == 10 and cub.species.tolist() == [1, 1, 2]
    train_idx, test_idx = zero_shot_split(cub.species, tmp_path / "xlsa17", tmp_path / "CUB_200_2011")
    assert train_idx.tolist() == [0, 1] and test_idx.tolist() == [2]
    assert len(dev_species(np.arange(1, 151), n_dev=30, seed=42)) == 30
