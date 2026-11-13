"""Small independent invariants for SC1's scientific calculations; CPU only."""
import math
import unittest

import r29rd_sc1 as rd
import numpy as np
import torch


def slow_ap(order, query_label, gallery_labels):
    if query_label < 0:
        return float('nan')
    labels = [gallery_labels[int(i)] for i in order if gallery_labels[int(i)] >= 0]
    positives = sum(label == query_label for label in labels)
    if not positives:
        return float('nan')
    hits = 0
    total = 0.0
    for rank, label in enumerate(labels, 1):
        if label == query_label:
            hits += 1
            total += hits / rank
    return total / positives


class MetricTests(unittest.TestCase):
    def test_unlabelled_and_no_positive(self):
        order = np.array([[0, 1, 2, 3, 4]] * 3)
        actual = rd.ranked_ap(order, np.array([1, -1, 7]), np.array([-1, 0, 1, -1, 1]))
        self.assertAlmostEqual(actual[0], (1 / 2 + 2 / 3) / 2, places=15)
        self.assertTrue(np.isnan(actual[1:]).all())

    def test_random_against_scalar_ap(self):
        rng = np.random.default_rng(492)
        labels = rng.integers(-1, 5, size=37)
        qlabels = np.arange(-1, 7)
        orders = np.stack([rng.permutation(37) for _ in qlabels])
        expected = [slow_ap(order, label, labels) for order, label in zip(orders, qlabels)]
        np.testing.assert_allclose(rd.ranked_ap(orders, qlabels, labels), expected, rtol=0, atol=2e-16, equal_nan=True)

    def test_ties_own_row_and_cross_modal_painting(self):
        # Equal scores force ascending gallery position; t2t self leaves gallery.
        vectors = dict(t=np.array([[1., 0.], [1., 0.], [1., 0.]], dtype=np.float32),
                       i=np.array([[1., 0.], [1., 0.]], dtype=np.float32))
        labels = {'emotion': dict(t=np.array([0, 1, 0]), i=np.array([0, 1]))}
        t2t = rd.score_direction(vectors, labels, 't2t', batch_size=2)['emotion']
        np.testing.assert_allclose(t2t, [.5, np.nan, 1.], equal_nan=True)
        t2i = rd.score_direction(vectors, labels, 't2i', batch_size=2)['emotion']
        np.testing.assert_allclose(t2i, [1., .5, 1.])
        i2t = rd.score_direction(vectors, labels, 'i2t', batch_size=2)['emotion']
        np.testing.assert_allclose(i2t, [(1. + 2. / 3.) / 2., .5])

    def test_query_subset_keeps_full_gallery(self):
        rng = np.random.default_rng(3)
        vectors = dict(t=rd.unit(rng.normal(size=(41, 7))), i=rd.unit(rng.normal(size=(13, 7))))
        labels = {'genre': dict(t=rng.integers(-1, 3, size=41), i=rng.integers(-1, 3, size=13))}
        for direction, side in [('t2i', 't'), ('i2t', 'i'), ('t2t', 't')]:
            full = rd.score_direction(vectors, labels, direction, batch_size=5)['genre']
            subset = rd.score_direction(vectors, labels, direction, query_limit=9, batch_size=4)['genre']
            positions = rd.selected_queries(len(vectors[side]), 9)
            np.testing.assert_allclose(subset, full[positions], rtol=0, atol=1e-15, equal_nan=True)

    def test_no_positive_count_and_all_emotion_classes(self):
        labels = dict(emotion=dict(t=np.array([0, 0, 1, 2]), i=np.array([0, 1])),
                      genre=dict(t=np.array([-1, 0, 0, 1]), i=np.array([0, -1])))
        counts = rd.finite_query_counts(dict(labels=labels))
        self.assertEqual(counts, {'emotion/t2i': 3, 'emotion/i2t': 2, 'emotion/t2t': 2,
                                  'genre/t2i': 2, 'genre/i2t': 1, 'genre/t2t': 2})

    def test_explicit_join_permutations_and_duplicates(self):
        np.testing.assert_array_equal(rd.join_rows([91, 5, 48], [48, 91]), [2, 0])
        with self.assertRaises(AssertionError):
            rd.join_rows([5, 5], [5])
        with self.assertRaises(KeyError):
            rd.join_rows([91, 5], [0])

    def test_tower_exact_gelu_and_output_normalisation(self):
        # Closed-form two-neuron calculation; dropout must be inactive.
        state = {'cap.trunk.0.weight': torch.tensor([[1., 0.], [0., 2.]]),
                 'cap.trunk.0.bias': torch.tensor([0.25, -0.5]),
                 'cap.output.weight': torch.tensor([[1., 2.], [-1., 1.], [0., 0.]]),
                 'cap.output.bias': torch.tensor([0., 0., 0.5])}
        x = rd.unit([[1., 2.], [-2., 1.]])
        expected = []
        for a, b in x:
            z = [float(a) + .25, 2 * float(b) - .5]
            h = [v * .5 * (1 + math.erf(v / math.sqrt(2))) for v in z]
            out = np.array([h[0] + 2 * h[1], -h[0] + h[1], .5])
            expected.append(out / np.linalg.norm(out))
        np.testing.assert_allclose(rd.tower(x, state, 'cap', batch_size=1), expected, atol=2e-7, rtol=0)

    def test_decisions_exact_ties(self):
        arms = {arm: {'group': .25} for arm in rd.ARMS}
        self.assertEqual(rd.decision(arms), ('siglip1', 'plain', 'plain'))
        arms['siglip1_prefix']['group'] = np.nextafter(.25, 1.)
        arms['p1_siglip1_prefix']['group'] = np.nextafter(.25, 1.)
        self.assertEqual(rd.decision(arms), ('siglip1_prefix', 'prefixed', 'prefixed'))


if __name__ == '__main__':
    unittest.main(verbosity=0)
