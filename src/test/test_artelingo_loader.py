import numpy as np
import pytest

from src.data.artelingo import join_annotations


def test_join_annotations_is_positional_by_sample_id():
    annotations = [{"emotion": "awe", "painting": "p0"}, {"emotion": "fear", "painting": "p1"}]
    emotions, paintings = join_annotations(np.array([1, 0]), annotations)
    assert emotions.tolist() == ["fear", "awe"] and paintings.tolist() == ["p1", "p0"]


@pytest.mark.parametrize("ids", [[0, 0], [0, 2], [-1, 0]])
def test_join_annotations_rejects_duplicate_or_out_of_range_ids(ids):
    annotations = [{"emotion": "awe", "painting": "p0"}, {"emotion": "fear", "painting": "p1"}]
    with pytest.raises(ValueError):
        join_annotations(np.array(ids), annotations)


def test_join_annotations_rejects_missing_or_empty_fields():
    with pytest.raises(ValueError):
        join_annotations(np.array([0]), [{"emotion": "awe"}])
    with pytest.raises(ValueError):
        join_annotations(np.array([0]), [{"emotion": "awe", "painting": ""}])


# ---- Final-review fix I5: art_style joined by the same positional join, one style per painting ----

import dataclasses  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

import src.data.artelingo as artelingo  # noqa: E402
from src.data.artelingo import ArtelingoData, join_art_styles  # noqa: E402

_ANNOTATIONS = [{"emotion": "awe", "painting": "p0", "art_style": "Baroque"},
                {"emotion": "fear", "painting": "p1", "art_style": "Cubism"},
                {"emotion": "sadness", "painting": "p0", "art_style": "Baroque"}]


def test_join_art_styles_is_positional_by_sample_id():
    styles = join_art_styles(np.array([1, 0, 2]), _ANNOTATIONS)
    assert styles.tolist() == ["Cubism", "Baroque", "Baroque"]


@pytest.mark.parametrize("ids", [[0, 0], [0, 3], [-1, 0]])
def test_join_art_styles_rejects_duplicate_or_out_of_range_ids(ids):
    with pytest.raises(ValueError):
        join_art_styles(np.array(ids), _ANNOTATIONS)


def test_join_art_styles_rejects_missing_empty_or_inconsistent_styles():
    with pytest.raises(ValueError):
        join_art_styles(np.array([0]), [{"emotion": "awe", "painting": "p0"}])
    with pytest.raises(ValueError):
        join_art_styles(np.array([0]), [{"emotion": "awe", "painting": "p0", "art_style": ""}])
    two_styles = [dict(_ANNOTATIONS[0]), dict(_ANNOTATIONS[2], art_style="Rococo")]   # p0 -> two styles
    with pytest.raises(ValueError, match="art_style"):
        join_art_styles(np.array([0, 1]), two_styles)


class _FakeFeatureManager:
    """Stands in for src.utils.FeatureManager: 3 rows stored in a shuffled sample-id order."""

    def __init__(self, storage_dir):
        self.total_samples = 3

    def load_all_to_ram(self, keys):
        return {key: torch.arange(6, dtype=torch.float32).reshape(3, 2) for key in keys}

    def get_all_sample_ids(self):
        return [2, 0, 1]


def test_load_artelingo_exposes_art_styles_by_the_same_positional_join(tmp_path, monkeypatch):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_ANNOTATIONS))
    monkeypatch.setattr(artelingo, "FeatureManager", _FakeFeatureManager)
    data = artelingo.load_artelingo(feature_dir="unused", annotations_path=path, expected_samples=3)
    assert "art_styles" in {f.name for f in dataclasses.fields(ArtelingoData)}
    assert data.sample_ids.tolist() == [2, 0, 1]
    assert data.paintings.tolist() == ["p0", "p0", "p1"]
    assert data.art_styles.tolist() == ["Baroque", "Baroque", "Cubism"]
    assert data.emotions.tolist() == ["sadness", "awe", "fear"]
