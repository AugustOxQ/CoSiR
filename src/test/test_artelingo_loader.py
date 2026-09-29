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
