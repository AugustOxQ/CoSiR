"""Stage 1c scientific invariants; no K* held data and no GPU access."""
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import r29rd_c1 as M
import r29rd_c1_core as C


class C1Tests(unittest.TestCase):
    def test_contingency_ami_matches_sklearn(self):
        rng = np.random.default_rng(12)
        for n in (7, 41, 400):
            y, p = rng.integers(0, 4, n), rng.integers(0, 5, n)
            table = np.bincount(y * 5 + p, minlength=20).reshape(4, 5)
            self.assertAlmostEqual(C.fast_ami(table), adjusted_mutual_info_score(y, p), places=12)

    def test_image_selection_guard_tie_and_no_expansion_priority(self):
        units = {a: .01 for a in M.ASPECTS}
        base = {a: .2 for a in M.ASPECTS}
        rows = {'none+k0': base, 'center+k0': {a:.21 for a in M.ASPECTS},
                'center+k10': {a:.210005 for a in M.ASPECTS},
                'whiten64+k0': {'emotion':.189, 'style':.5, 'genre':.5}}
        pick, _, kept = C.select_image(rows, units)
        self.assertEqual(pick, 'center+k0')
        self.assertFalse(kept['whiten64+k0'])
        rows['center+k0'] = {a:.200005 for a in M.ASPECTS}
        rows['center+k10'] = {a:.200005 for a in M.ASPECTS}
        self.assertEqual(C.select_image(rows, units)[0], 'none+k0')

    def test_image_menu(self):
        self.assertEqual(len(C.menu(2048)), 25)
        self.assertEqual(len(C.menu(6144)), 25)
        self.assertEqual(len(C.menu(64)), 15)

    def test_pca_whitening_cosines_against_direct_eigendecomposition(self):
        rng=np.random.default_rng(931)
        fit=C.unit(rng.normal(size=(31,7)))
        queries=C.unit(rng.normal(size=(9,7)))
        values,vectors=np.linalg.eigh(np.cov(fit.astype(np.float64),rowvar=False))
        order=np.argsort(values)[::-1][:4]
        transform=(vectors[:,order]/np.sqrt(values[order]+1e-6*values.max())).astype(np.float32)
        expected=C.unit((queries-C.means(fit))@transform)
        actual=C.apply_map(queries,C.mapper(fit,'whiten4'))
        np.testing.assert_allclose(actual@actual.T,expected@expected.T,rtol=0,atol=3e-7)

    def test_bootstrap_query_transcript_by_hand(self):
        candidate = np.arange(30, dtype=np.float64).reshape(5,6)/50
        comparator = np.arange(6, dtype=np.float64)/30
        rng = np.random.default_rng(0)
        values = []
        for _ in range(19):
            I = rng.integers(0,6,size=6)
            S = rng.choice([0,1,2,3,4],size=5,replace=True)
            values.append(candidate[S][:,I].mean() - comparator[I].mean())
        actual = C.retrieval_interval(candidate, comparator, draws=19)
        np.testing.assert_allclose([actual['lo'],actual['hi']],np.percentile(values,[2.5,97.5]),rtol=0,atol=2e-16)

    def test_bootstrap_painting_transcript_unequal_caption_counts(self):
        pid = np.array([0,0,1,2,2,2])
        candidate = np.arange(30,dtype=np.float64).reshape(5,6)/50
        comparator = np.arange(6,dtype=np.float64)/30
        candidate[:,2] = np.nan
        comparator[2] = np.nan
        # Only paintings 0 and 2 have scored queries; population is sorted [0,2].
        rng = np.random.default_rng(0)
        values = []
        for _ in range(19):
            I = rng.integers(0,2,size=2)
            S = rng.choice([0,1,2,3,4],size=5,replace=True)
            positions = np.concatenate([np.flatnonzero(pid == [0,2][i]) for i in I])
            values.append(candidate[S][:,positions].mean() - comparator[positions].mean())
        actual = C.retrieval_interval(candidate,comparator,pid,draws=19)
        np.testing.assert_allclose([actual['lo'],actual['hi']],np.percentile(values,[2.5,97.5]),rtol=0,atol=2e-16)
        self.assertEqual(actual['n_queries'],5)
        self.assertEqual(actual['n_units'],2)

    def test_ami_rows_then_seeds_transcript(self):
        rng = np.random.default_rng(7)
        y = np.array([0,0,-1,1,1,0])
        candidate = rng.integers(0,2,size=(5,3,len(y)))
        comparator = rng.integers(0,2,size=(3,len(y)))
        rng = np.random.default_rng(0)
        expected=[]
        for _ in range(11):
            I=rng.integers(0,len(y),size=len(y))
            S=rng.choice(5,size=5,replace=True)
            good=y[I]>=0
            yy=y[I][good]
            cv=np.mean([np.mean([adjusted_mutual_info_score(yy,p[I][good]) for p in candidate[s]]) for s in S])
            bv=np.mean([adjusted_mutual_info_score(yy,p[I][good]) for p in comparator])
            expected.append(cv-bv)
        actual=C.ami_interval(candidate,comparator,y,draws=11,workers=1)
        np.testing.assert_allclose([actual['lo'],actual['hi']],np.percentile(expected,[2.5,97.5]),rtol=0,atol=1e-12)

    def test_sampled_replay_does_not_certify_a_full_cell(self):
        from types import SimpleNamespace
        args=SimpleNamespace(family='text',aspects=['emotion'],query_limit=128,output=M.HERE/'unused_replay_test.json')
        row=dict(id='text/emotion/t2i/plain',full_interval=True,full_scoring=False,
                 max_abs_diff=1e-16,scoring_max_abs_diff=1e-5)
        with patch.object(M,'replay_text',return_value=[row]), patch.object(M.rd,'write_json') as output, redirect_stdout(io.StringIO()):
            M.replays(args)
        report=output.call_args.args[1]['replays']['text']
        self.assertTrue(report['interval_gate_1e9'])
        self.assertFalse(report['scoring_gate_1e9'])

    def test_crossfit_parity_is_defined_before_label_masking(self):
        x=C.unit(np.arange(1,25,dtype=np.float32).reshape(8,3))
        labels=np.array([-1,0,1,0,0,-1,1,0])
        fits=[]
        def spy(fit,w):
            fits.append(fit.copy())
            return None,None
        with patch.object(C,'mapper',side_effect=spy):
            C.crossfit(x,labels,['none+k0'],batch_size=3)
        self.assertEqual([len(fit) for fit in fits],[4,4])
        np.testing.assert_allclose(fits[0],C.unit(x)[1::2])
        np.testing.assert_allclose(fits[1],C.unit(x)[0::2])

    def test_held_access_guard_before_read(self):
        with patch.object(M,'export',side_effect=AssertionError('must not read')):
            with self.assertRaises(RuntimeError):
                M.candidate_images('held','emotion',0,{})

    def test_ladder_and_inconclusive_boundaries(self):
        rows=[dict(id=f'x/{i}/p1',outcome='WIN' if i<12 else 'BELOW') for i in range(15)]
        rows += [dict(id=f'x/{i}/q1',outcome='LEVEL' if i<12 else 'BELOW') for i in range(15)]
        self.assertEqual(C.ladder(rows)['level'],1)
        rows[11]['outcome']='BELOW'
        self.assertEqual(C.ladder(rows)['level'],0)
        rows[11]['outcome']='WIN'
        for r in rows[:15]:r['outcome']='WIN'
        self.assertEqual(C.ladder(rows)['level'],3)
        rows[26]['outcome']='BELOW'
        self.assertEqual(C.ladder(rows)['level'],2)
        rows[26]['outcome']='LEVEL'
        rows[14]['outcome']='INCONCLUSIVE'
        actual=C.ladder(rows)
        self.assertEqual(actual['level'],1)
        self.assertTrue(actual['inconclusive_above'])
        self.assertEqual(C.outcome(0.,.1),'LEVEL')
        self.assertEqual(C.outcome(-.1,.1),'BELOW')

    def test_text_own_row_and_unlabelled_gallery(self):
        x=C.unit([[1,0],[1,0],[1,0],[1,0]])
        labels=dict(t=np.array([0,-1,1,0]),i=np.array([0,1]))
        aps,_=C.text_ap(x,x[:2],labels,'t2t',menu=['r0'],batch_size=2)
        np.testing.assert_allclose(aps['r0'],[.5,np.nan,np.nan,1.],equal_nan=True)

    def test_untold_join_order_and_norm(self):
        heads={a:np.eye(3,dtype=np.float16)*(i+1) for i,a in enumerate(M.ASPECTS)}
        joined=M.untold(heads)
        self.assertEqual(joined.shape,(3,9))
        np.testing.assert_allclose(np.linalg.norm(joined,axis=1),1.,atol=1e-7)


if __name__=='__main__':
    unittest.main(verbosity=0)
