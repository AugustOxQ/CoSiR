"""GoEmotions affect extraction (affect spec §4); skipped when the model is not in the local HF cache."""

import numpy as np
import pytest

from src.data.affect import GOEMOTIONS_NUM_LABELS, goemotions_probabilities, load_goemotions

TEXTS = ["I am so happy and grateful, this is wonderful!",
         "This is heartbreaking, I feel so sad and lonely.",
         "A bowl of pears on a wooden table."]


@pytest.fixture(scope="module")
def loaded():
    try:
        return load_goemotions(device="cpu")
    except OSError as err:                                  # not cached on this machine
        pytest.skip(f"GoEmotions model not in the local HF cache: {err}")


def test_shape_range_and_dtype(loaded):
    p = goemotions_probabilities(TEXTS, loaded=loaded)
    assert p.shape == (3, GOEMOTIONS_NUM_LABELS) and p.dtype == np.float32
    assert (p >= 0).all() and (p <= 1).all()


def test_joy_and_sadness_are_read_correctly(loaded):
    _, model = loaded
    index = {name: int(i) for i, name in model.config.id2label.items()}
    p = goemotions_probabilities(TEXTS, loaded=loaded)
    assert p[0, index["joy"]] > p[0, index["sadness"]]
    assert p[1, index["sadness"]] > p[1, index["joy"]]


def test_batch_size_does_not_change_the_output(loaded):
    one = goemotions_probabilities(TEXTS, loaded=loaded, batch_size=1)
    all_at_once = goemotions_probabilities(TEXTS, loaded=loaded, batch_size=3)
    assert np.allclose(one, all_at_once, atol=1e-5)


def test_empty_input_and_bad_arguments(loaded):
    assert goemotions_probabilities([], loaded=loaded).shape == (0, GOEMOTIONS_NUM_LABELS)
    with pytest.raises(ValueError, match="batch_size"):
        goemotions_probabilities(TEXTS, loaded=loaded, batch_size=0)
    with pytest.raises(TypeError, match="str"):
        goemotions_probabilities(["fine", 3], loaded=loaded)
