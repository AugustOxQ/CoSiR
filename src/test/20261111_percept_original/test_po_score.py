"""r20 / rclass / own readings against direct scikit-learn on a tiny synthetic case."""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import po_score as S


def direct_kmeans(prob, k, state):
    p = prob.astype(np.float32)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    return MiniBatchKMeans(n_clusters=k, random_state=state, n_init=3, batch_size=4096).fit(p).labels_


def case(n=240, k=6):
    rng = np.random.default_rng(5)
    prob = rng.uniform(.02, .98, size=(n, k)).astype(np.float32)
    prob[7] = .5                                                  # a row with all-equal maxima: own = lowest index
    labels = {a: rng.integers(0, c, size=n).astype(np.int64) for a, c in (('emotion', 3), ('style', 4), ('genre', 2))}
    labels['genre'][::3] = -1
    return prob, labels, dict(emotion=3, style=4, genre=2)


def test_readings_match_direct_sklearn():
    prob, labels, counts = case()
    got = S.score_probs(prob, labels, counts)
    for aspect, y in labels.items():
        keep = y >= 0
        want20 = [adjusted_mutual_info_score(y[keep], direct_kmeans(prob, 20, st)[keep]) for st in (0, 1, 2)]
        wantc = [adjusted_mutual_info_score(y[keep], direct_kmeans(prob, counts[aspect], st)[keep]) for st in (0, 1, 2)]
        assert np.allclose(got['r20'][aspect], want20, atol=1e-12, rtol=0)
        assert np.allclose(got['rclass'][aspect], wantc, atol=1e-12, rtol=0)
        own = prob.argmax(axis=1)
        assert np.allclose(got['own'][aspect], [adjusted_mutual_info_score(y[keep], own[keep])], atol=1e-12, rtol=0)
    assert prob.argmax(axis=1)[7] == 0


def test_aggregate_leaves_out_collapsed_seeds():
    def res(c, v):
        vals = {r: {a: dict(per_state=[v], mean=v) for a in S.P.ASPECTS} for r in S.READINGS}
        return dict(collapsed=c, values=vals)
    table, used = S.aggregate({0: res(False, .1), 1: res(True, .9), 2: res(False, .3)})
    assert used == [0, 2] and abs(table['rclass']['emotion'] - .2) < 1e-12
    table, used = S.aggregate({0: res(True, .5)})
    assert used == [] and table['own']['style'] is None
