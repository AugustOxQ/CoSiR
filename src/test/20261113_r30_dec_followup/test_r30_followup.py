import numpy as np
import r30_followup as r


def test_mapping_and_ties():
    names = ['sadness', 'awe', 'something', 'anger', 'amusement', 'fear', 'contentment', 'disgust', 'excitement']
    dist = np.zeros((3, 9))
    dist[0, [1, 4]] = [.3, .7]
    dist[1, [0, 2]] = [.5, .5]
    dist[2, 2] = 1
    shares, labels = r.shares_and_labels(dist, names)
    np.testing.assert_allclose(shares, [[1, 0, 0], [0, .5, .5], [0, 0, 1]])
    assert labels.tolist() == [0, -1, 2]


def test_gap():
    shares = np.array([[.1, .9, 0], [.3, .7, 0], [.8, .2, 0]])
    assert np.isclose(r.pos_gap(shares, np.array([0, 0, 1])), .6)
