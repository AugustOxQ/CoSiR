import numpy as np

from src.data.artelingo_splits import encode_labels


def test_encode_labels_missing_and_excluded():
    codes, names = encode_labels(np.array(["sad", "", "awe", "something else", None, "sad"], dtype=object),
                                 exclude=("something else",))
    assert names == ["awe", "sad"]
    assert codes.tolist() == [1, -1, 0, -1, -1, 1]
