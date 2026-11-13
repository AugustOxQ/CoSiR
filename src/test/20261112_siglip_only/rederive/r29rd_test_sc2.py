"""Independent checks of SC2 boundaries, masking populations and forward diagnostics."""
import unittest
from unittest.mock import patch

import r29rd_sc2 as rd2
import numpy as np


class SC2Tests(unittest.TestCase):
    def test_exact_level_boundaries(self):
        baseline = dict(image=dict(m=.4, r=.01, best=.405, lower=.39),
                        text=dict(m=.3, r=.02, best=.31, lower=.28))
        self.assertEqual(rd2.level_for(.39, .28, baseline), 1)
        self.assertEqual(rd2.level_for(.39, .30, baseline), 2)
        self.assertEqual(rd2.level_for(.405, .31, baseline), 2)
        self.assertEqual(rd2.level_for(np.nextafter(.405, 1.), np.nextafter(.31, 1.), baseline), 3)
        self.assertEqual(rd2.level_for(np.nextafter(.39, 0.), .40, baseline), 0)
        self.assertEqual(rd2.level_for(.50, np.nextafter(.28, 0.), baseline), 0)
        self.assertEqual(rd2.level_for(.50, np.nextafter(.30, 0.), baseline), 1)

    def test_ami_clusters_unlabelled_rows_without_normalising(self):
        raw = np.array([[10, 0], [0, 3], [4, 5], [2, 1], [1, 3], [9, 2]], dtype=np.float16)
        labels = np.array([0, 1, -1, 0, 1, -1])
        calls = []

        class FakeKMeans:
            def __init__(self, **params):
                calls.append(params)

            def fit_predict(self, x):
                np.testing.assert_array_equal(x, raw.astype(np.float32))
                self_dtype = x.dtype
                assert self_dtype == np.float32 and len(x) == 6
                return np.array([1, 0, 0, 1, 0, 1])

        with patch.object(rd2, 'MiniBatchKMeans', FakeKMeans):
            score, states = rd2.ami_cell(raw, labels, 2)
        self.assertEqual(score, 1.)
        self.assertEqual(states, [1., 1., 1.])
        self.assertEqual(calls, [dict(n_clusters=2, n_init=3, batch_size=4096, random_state=i) for i in range(3)])

    def test_image_map_masks_unlabelled_and_removes_self(self):
        # Every labelled vector ties; ascending positions determine AP.
        raw = np.array([[2, 0], [3, 0], [9, 1], [4, 0]], dtype=np.float16)
        labels = np.array([0, 1, -1, 0])
        score, count = rd2.image_map(raw, labels, batch_size=2)
        # Class-1 singleton query is dropped; class-0 APs are 1/2 and 1.
        self.assertEqual(count, 2)
        self.assertEqual(score, .75)

    def test_image_map_against_scalar_reference(self):
        raw = np.array([[1, 0], [1, 1], [0, 2], [-1, 0], [2, -1], [3, 4]], dtype=np.float16)
        labels = np.array([0, 0, 1, 1, -1, 2])
        x = rd2.rd.unit(raw[labels >= 0])
        lab = labels[labels >= 0]
        expected = []
        for own in range(len(x)):
            order = sorted((i for i in range(len(x)) if i != own), key=lambda i: (-float(x[own] @ x[i]), i))
            hit_positions = [j + 1 for j, i in enumerate(order) if lab[i] == lab[own]]
            if hit_positions:
                expected.append(sum(hits / rank for hits, rank in enumerate(hit_positions, 1)) / len(hit_positions))
        score, count = rd2.image_map(raw, labels, batch_size=3)
        self.assertEqual(count, len(expected))
        self.assertAlmostEqual(score, sum(expected) / len(expected), places=15)

    def test_quantised_forward_diagnostic(self):
        first = rd2.forward_comparison(np.array([[1.0001, .5003]], dtype=np.float32),
                                      np.array([[1., .5]], dtype=np.float16))
        self.assertEqual(first['share_equal'], .5)
        self.assertEqual(first['max_abs_diff'], .00048828125)
        self.assertEqual(first['equal_rows'], 0)
        second = rd2.forward_comparison(np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float16))
        combined = rd2.forward_summary(dict(first=first, second=second))
        self.assertEqual(combined['elements'], 6)
        self.assertEqual(combined['share_equal'], 5 / 6)
        self.assertEqual(combined['max_abs_diff'], .00048828125)

    def test_baseline_reads_frozen_group_mean(self):
        baseline, cells, per_seed = rd2.frozen_baseline()
        self.assertEqual(baseline['image']['m'], .38629170813482466)
        self.assertEqual(f'{baseline["text"]["m"]:.10f}', '0.2593512884')
        self.assertEqual(f'{baseline["text"]["r"]:.10f}', '0.0020107136')
        self.assertEqual(baseline['image']['best'], max(per_seed['image']))
        self.assertEqual(len(cells['image']), 6)
        self.assertEqual(len(cells['text']), 9)


if __name__ == '__main__':
    unittest.main(verbosity=0)
