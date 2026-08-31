"""Focused tests for browser-only A/B/C/D example selection."""
import importlib.util
from pathlib import Path

import numpy as np


_GENERATOR = Path(__file__).with_name("generate_abcd_examples.py")
_SPEC = importlib.util.spec_from_file_location("generate_abcd_examples", _GENERATOR)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_choose_indices_uses_unique_hubs_and_cd_pairs():
    """Several hubs can expose one C/D pair; browser sampling must not spend its
    whole unconnected bucket on that same visual contrast."""
    labels = np.array([
        "unconnected", "unconnected", "unconnected",
        "txt_only", "txt_only", "img_only", "img_only", "both", "both",
    ])
    hub = np.arange(10, 19)
    c = np.array([1, 1, 4, 6, 8, 10, 12, 14, 16])
    d = np.array([2, 2, 5, 7, 9, 11, 13, 15, 17])
    distance = np.array([0.9, 0.8, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    selected = _MODULE._choose_indices(
        labels, hub, c, d, distance, np.ones(len(labels), dtype=bool),
        examples_per_bucket=2, rng=np.random.default_rng(0),
    )

    assert selected[:2] == [0, 2], selected


def test_annotations_follow_feature_row_sample_ids_not_row_positions():
    """Feature-store rows are shuffled, so graph row 0 must resolve through its
    sample ID before the browser reads a caption or image."""
    annotations = [
        {"caption": "row zero", "image": "zero.jpg", "image_id": "zero"},
        {"caption": "row one", "image": "one.jpg", "image_id": "one"},
        {"caption": "row two", "image": "two.jpg", "image_id": "two"},
    ]

    by_position = _MODULE._annotations_by_feature_position(annotations, [2, 0, 1])

    assert [item["sample_id"] for item in by_position] == [2, 0, 1]
    assert [item["caption"] for item in by_position] == ["row two", "row zero", "row one"]


if __name__ == "__main__":
    test_choose_indices_uses_unique_hubs_and_cd_pairs()
    test_annotations_follow_feature_row_sample_ids_not_row_positions()
    print("ALL TESTS PASSED")
