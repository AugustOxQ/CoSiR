"""Tiny deterministic fixtures; no stage-2 dataset/model access."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import r29rd_s2e as E


class RuleFixtures(unittest.TestCase):
    def test_strict_outcome_boundaries(self):
        self.assertEqual(E.C.outcome(1e-15, .1),'WIN')
        self.assertEqual(E.C.outcome(0., .1),'LEVEL')
        self.assertEqual(E.C.outcome(-.1, .1),'BELOW')
        self.assertEqual(E.C.outcome(0., 0.),'BELOW')

    def test_sc9_ties_and_paired_seed_ahead(self):
        k=[.1,.3,.2,.4,.5]
        d=E.sc9(dict(siglip2=k,clip_l14=[.1]*5),k,.99)
        self.assertEqual(d['pick'],'siglip1')
        self.assertEqual(d['runner_up'],'siglip2')
        self.assertEqual(d['seeds_ahead'],[])
        self.assertTrue(d['close_call'])
        self.assertNotIn('qwen',d['means'])
        d=E.sc9(dict(siglip2=[.5,.2,.3,.5,.6],clip_l14=[.1]*5),k,.99)
        self.assertEqual(d['seeds_ahead'],[0,2,3,4])
        self.assertEqual(d['n_seeds_ahead'],4)

    def test_ladder_inconclusive(self):
        rows=[dict(side=s,outcome='WIN') for s in ('plain','q1') for _ in range(6)]
        self.assertEqual(E.ladder(rows)['level'],3)
        rows[0]['outcome']='INCONCLUSIVE'
        r=E.ladder(rows)
        self.assertEqual(r['level'],1)
        self.assertTrue(r['inconclusive_above'])
        rows[1]['outcome']='BELOW'
        self.assertEqual(E.ladder(rows)['level'],0)

    def test_held_access_guard(self):
        with patch.object(E,'ALLOW_HELD',False):
            with self.assertRaisesRegex(RuntimeError,'controller-only'):
                E.export_path('siglip2','held','emotion',0)
            self.assertIn('cosir_r14',str(E.export_path('siglip2','held','emotion',0,True)))
            with self.assertRaisesRegex(ValueError,'picks-committed'):
                E.do_held(SimpleNamespace(picks_committed=False))

    def test_output_confinement(self):
        for p in (E.HERE/'../forbidden.json',E.HERE/'other_job.json'):
            with self.assertRaises(ValueError):E.output_path(p)
        self.assertEqual(E.output_path(E.HERE/'s2e_val.json').name,'s2e_val.json')

    def test_untold_head_order(self):
        heads={a:np.asarray([[i+1.,1.],[1.,i+2.]],dtype=np.float16) for i,a in enumerate(E.ASPECTS)}
        actual=E.untold(heads)
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1.,atol=1e-6)
        np.testing.assert_allclose(actual[:,:2]*np.sqrt(3),E.C.unit(heads['emotion']),atol=1e-7)

    def test_ami_repeated_rows_missing_label(self):
        y=np.asarray([0,0,1,1,-1,2,2])
        p=np.asarray([[0,0,0,1,1,2,2],[2,2,1,1,0,0,0],[0,1,0,1,0,2,2]])
        ids=np.asarray([0,0,2,4,5,6,6])
        weights=np.bincount(ids,minlength=len(y))
        expected=np.mean([adjusted_mutual_info_score(y[ids][y[ids]>=0],z[ids][y[ids]>=0]) for z in p])
        self.assertAlmostEqual(E.C.part_value(p,y,weights),expected,12)

    def test_retrieval_rng_draw_order(self):
        c=np.arange(35,dtype=np.float64).reshape(5,7)/41
        b=np.arange(7,dtype=np.float64)/50
        rng=np.random.default_rng(0)
        boot=[]
        for _ in range(101):
            ids=rng.integers(0,7,size=7)
            seeds=rng.choice([0,1,2,3,4],size=5,replace=True)
            boot.append(c[seeds][:,ids].mean()-b[ids].mean())
        result=E.C.retrieval_interval(c,b,draws=101)
        np.testing.assert_allclose([result['lo'],result['hi']],np.percentile(boot,[2.5,97.5]),atol=1e-15)

    def test_ap_self_exclusion_stable_ties(self):
        # All cosine ties: gallery order 0,1,2,3, excluding the query.
        x=np.ones((4,2),dtype=np.float32)/np.sqrt(np.float32(2))
        labels=np.asarray([0,0,1,1])
        ap=E.ranked_maps(x,labels,(0,))[0]
        np.testing.assert_allclose(ap,[1.,1.,1/3,1/3],atol=1e-15)

    def test_readout_float64_normalization_single_cast(self):
        x=np.asarray([[1.,2.,3.],[3.,2.,1.],[2.,4.,1.]],dtype=np.float32)
        fit=E.map_fit(x,'center')
        expected=x.astype(np.float64)-x.astype(np.float64).mean(0)
        expected=(expected/np.linalg.norm(expected,axis=1,keepdims=True)).astype(np.float32)
        np.testing.assert_array_equal(E.map_apply(x,fit),expected)

    def test_crossfit_parity_and_missing_gallery_labels(self):
        # Every parity half contains two examples per class, plus one unlabelled.
        x=np.asarray([[1.,0.],[1.,.01],[1.,.02],[1.,.03],
                      [0.,1.],[.01,1.],[.02,1.],[.03,1.],[.3,.3],[.3,.4]],dtype=np.float32)
        y=np.asarray([0,0,0,0,1,1,1,1,-1,-1])
        args=SimpleNamespace(query_limit=0,batch_size=3)
        out=E.crossfit(E.C.unit(x),{'a':y},['none+k0'],args)
        self.assertAlmostEqual(out['a']['none+k0'],1.)

    def test_sampled_ranking_cannot_certify_held(self):
        cell=dict(id='siglip2/map/emotion/plain',full_interval=True,full_scoring=False)
        rows=[dict(key='frozen/'+k,passed=True) for k in ('value','comparator_value','d','lo','hi','r')]
        report=dict(cells=[cell],replays=rows)
        with patch.object(E,'target',return_value=(Path('/frozen.json'),'frozen',{},{})):
            self.assertFalse(E.certified([report],'siglip2','emotion','map','plain'))
            cell['full_scoring']=True
            self.assertTrue(E.certified([report],'siglip2','emotion','map','plain'))
            rows[0]['passed']=False
            self.assertFalse(E.certified([report],'siglip2','emotion','map','plain'))

    def test_export_emb_key_and_row_hash(self):
        names=['painting_a','painting_b']
        x=np.zeros((2,2048),dtype=np.float16)
        x[:,0]=1
        info=dict(list_sha256=E.names_hash(names),aspect='emotion',seed=0)
        with patch.object(E,'frame',return_value=dict(names=names)), \
             patch.object(E,'read_json',return_value=info), \
             patch.object(E,'track',side_effect=Path), \
             patch.object(E.np,'load',return_value=nullcontext({'emb':x})) as opened:
            np.testing.assert_array_equal(E.export('siglip2','val','emotion',0),x)
            info['list_sha256']='wrong-row-order'
            with self.assertRaisesRegex(AssertionError,'row order mismatch'):
                E.export('siglip2','val','emotion',0)
            self.assertEqual(opened.call_count,1)

    def test_readout_guard_and_tie_priority(self):
        r={a:.1 for a in E.ASPECTS}
        base={a:.5 for a in E.ASPECTS}
        rows={'none+k0':base,'center+k0':{a:.50001 for a in E.ASPECTS},
              'center+k10':dict(emotion=.39,style=.9,genre=.9)}
        pick,_,kept=E.C.select_image(rows,r)
        self.assertEqual(pick,'none+k0')
        self.assertFalse(kept['center+k10'])

if __name__=='__main__':unittest.main()
