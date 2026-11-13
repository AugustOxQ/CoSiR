"""In-memory fixtures only; no r29/r28 model export is opened."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import r29rd_s2abl as S


class Numerics(unittest.TestCase):
    def test_weighted_ami_equals_literal_resampling(self):
        y = np.array([0, 0, -1, 1, 1, 2, 2, 2])
        rng = np.random.default_rng(6)
        p = rng.integers(0, 3, size=(5, 3, len(y)))
        rows = np.array([0, 1, 2, 2, 3, 3, 5, 7])
        seeds = np.array([4, 4, 0, 1, 2])
        yy = y[rows]
        expected = np.mean([np.mean([adjusted_mutual_info_score(yy[yy >= 0], part[rows][yy >= 0])
                                     for part in p[s]]) for s in seeds])
        actual = S.ami_weighted(p, y, np.bincount(rows, minlength=len(y)), seeds)
        self.assertAlmostEqual(expected, actual, places=12)

    def test_empty_and_single_class_ami(self):
        self.assertEqual(S.fast_ami(np.zeros((3, 3), dtype=int)), 1.)
        self.assertEqual(S.fast_ami(np.array([[3]])), 1.)

    def test_points_mask_unknown_after_partition(self):
        y = np.array([0, 0, -1, 1, 1])
        parts = np.array([[[0, 0, 2, 1, 1]]*3]*5)
        np.testing.assert_array_equal(S.ami_points(parts, y), np.ones(5))

    def test_kmeans_receives_every_row_without_normalising(self):
        x = np.array([[2., 0.], [0., 3.], [4., 4.]])
        seen = []
        class KM:
            def __init__(self, **kw):
                seen.append(kw)
            def fit_predict(self, rows):
                np.testing.assert_array_equal(rows, x.astype(np.float32))
                return np.array([0, 1, 1])
        with patch.object(S, 'MiniBatchKMeans', KM):
            self.assertEqual(S.partitions(x, 2).shape, (3, 3))
        self.assertEqual([v['random_state'] for v in seen], [0, 1, 2])
        self.assertTrue(all(v['n_init'] == 3 and v['batch_size'] == 4096 for v in seen))

    def test_map_masks_unknown_excludes_self_and_drops_singletons(self):
        x = np.array([[1., 0.], [2., 0.], [0., 3.], [0., 9.], [0., -4.]])
        ap, qids = S.map_ap(x, np.array([0, 0, 1, -1, 2]), batch_size=2)
        np.testing.assert_array_equal(qids, [0, 1, 2, 4])
        np.testing.assert_array_equal(ap[:2], [1., 1.])
        self.assertTrue(np.isnan(ap[2:]).all())

    def test_map_rank_average_precision_hand_fixture(self):
        # For q0: q1 relevant at rank 1, q2 irrelevant rank 2, q3 relevant rank 3.
        x = np.array([[1., 0.], [1., .1], [1., .2], [1., .3]], dtype=np.float32)
        ap, _ = S.map_ap(x, np.array([0, 0, 1, 0]))
        self.assertAlmostEqual(ap[0], (1.+2./3.)/2.)

    def test_shared_map_bootstrap_matches_literal_draw_order(self):
        rng = np.random.default_rng(5)
        k, r = rng.random((5, 7)), rng.random((5, 7))
        k[:, 3] = r[:, 3] = np.nan
        expected, gen = [], np.random.default_rng(0)
        for _ in range(141):
            q = gen.integers(0, 6, size=6)
            s = gen.choice(5, size=5, replace=True)
            expected.append((r[:, [0, 1, 2, 4, 5, 6]][s][:, q]
                             - k[:, [0, 1, 2, 4, 5, 6]][s][:, q]).mean())
        np.testing.assert_allclose(S.map_interval(r, k, 141), np.percentile(expected, [2.5, 97.5]), atol=1e-15)

    def test_shared_ami_bootstrap_matches_literal_draw_order(self):
        gen = np.random.default_rng(7)
        r, k = gen.integers(0, 3, size=(2, 5, 3, 9))
        y = np.array([0, 0, 0, 1, -1, 1, 2, 2, 2])
        gen, expected = np.random.default_rng(0), []
        for _ in range(12):
            rows = gen.integers(0, len(y), size=len(y))
            s = gen.choice(5, size=5, replace=True)
            yy = y[rows]
            def arm(parts):
                return np.mean([np.mean([adjusted_mutual_info_score(yy[yy >= 0], p[rows][yy >= 0]) for p in parts[i]]) for i in s])
            expected.append(arm(r)-arm(k))
        np.testing.assert_allclose(S.ami_interval(r, k, y, 12, 1), np.percentile(expected, [2.5, 97.5]), atol=1e-12)

    def test_equal_arms_have_zero_intervals(self):
        a = np.arange(5*7, dtype=float).reshape(5, 7)
        self.assertEqual(S.map_interval(a, a, 15), (0., 0.))
        p = np.tile(np.arange(7) % 2, (5, 3, 1))
        self.assertEqual(S.ami_interval(p, p, np.arange(7) % 3, 5, 1), (0., 0.))

    def test_mismatched_query_masks_rejected(self):
        k = np.ones((5, 7)); r = k.copy(); r[0, 0] = np.nan
        with self.assertRaises(AssertionError):
            S.map_interval(r, k, 1)

    def test_held_guard_precedes_model_reads(self):
        import argparse
        args = argparse.Namespace(mode='held', picks_committed=False, seeds=list(range(5)))
        with patch.object(S, 'track'), patch.object(S, 'load_export') as load:
            with self.assertRaisesRegex(ValueError, 'picks-committed'):
                S.run(args)
            load.assert_not_called()

    def test_no_replay_path_is_r29(self):
        self.assertNotIn('cosir_r29', str(S.R14))
        self.assertNotIn('cosir_r29', str(S.R28))

    def test_outputs_confined_to_this_job(self):
        for p in ['/tmp/r29rd_s2abl.json', str(S.HERE/'other_job.json')]:
            with self.assertRaises(ValueError):
                S.writable(p)
        self.assertEqual(S.writable(S.HERE/'s2abl_held.json'), S.HERE/'s2abl_held.json')

    def test_replay_target_values_are_compare_only(self):
        d = S.compare(Path('frozen.json'), 'x', .1, .10000000001, 'ami')
        self.assertTrue(d['pass_'])
        self.assertGreater(d['abs_diff'], 0.)

    def test_export_emb_key_and_row_identity(self):
        names = ['painting_a', 'painting_b']
        metadata = dict(aspect='emotion', seed=0, list_sha256=S.names_hash(names))
        archive = {'emb': np.ones((2, 2048), dtype=np.float16)}
        with patch.object(S, 'read_json', return_value=metadata), patch.object(S, 'track', side_effect=Path), \
                patch.object(S.np, 'load') as load:
            load.return_value.__enter__.return_value = archive
            x = S.load_export(Path('fixture.npz'), names, 'emotion', 0)
            self.assertEqual(x.dtype, np.float32)
            np.testing.assert_array_equal(x, archive['emb'])

    def test_export_wrong_row_identity_prevents_array_read(self):
        with patch.object(S, 'read_json', return_value={'list_sha256': 'wrong'}), \
                patch.object(S.np, 'load') as load:
            with self.assertRaises(AssertionError):
                S.load_export(Path('fixture.npz'), ['a', 'b'], 'emotion', 0)
            load.assert_not_called()

    def test_c1_missing_seed_targets_are_explicit(self):
        y = np.array([0, 0, 1, 1])
        p = np.tile(y, (3, 1))
        arms = {'kstar': {'emotion': [{'parts': p} for _ in range(5)]}}
        record = {'cells': [{'id': 'ami/emotion/p1', 'value': 1.}]}
        with patch.object(S, 'read_json', return_value=record), patch.object(Path, 'glob', return_value=[]):
            result = S.c1_check([], arms, {'emotion': y}, ['emotion'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['mean_checks_pass'])
        self.assertEqual(result['kstar_per_seed']['emotion'], [1.]*5)
        self.assertEqual(len(result['missing']), 5)


if __name__ == '__main__':
    unittest.main()
