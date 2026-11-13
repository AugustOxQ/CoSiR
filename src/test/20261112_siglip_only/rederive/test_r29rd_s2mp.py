"""Small independent protocol fixtures, no external data or stage-2 held reads."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import r29rd_s2mp as m


class ProtocolTests(unittest.TestCase):
    def test_ami_repeated_rows_and_missing_labels(self):
        y=np.array([0,1,0,1,-1,2,2])
        p=np.array([[1,0,1,2,1,2,0],[0,0,2,1,1,2,2],[1,1,0,0,0,2,2]])
        w=np.array([1,3,0,1,5,2,1])
        idx=np.repeat(np.arange(len(y)),w)
        idx=idx[y[idx]>=0]
        expected=np.mean([adjusted_mutual_info_score(y[idx],row[idx]) for row in p])
        self.assertAlmostEqual(m.ami(p,y,w),expected,places=11)

    def test_ami_matches_sklearn_random_contingencies(self):
        rng=np.random.default_rng(42)
        for n in (8,31,200):
            y=rng.integers(0,4,n)
            p=rng.integers(0,5,(3,n))
            self.assertAlmostEqual(m.ami(p,y),np.mean([adjusted_mutual_info_score(y,z) for z in p]),places=11)

    def test_bootstrap_transcript(self):
        rng=np.random.default_rng(1)
        c=rng.random((5,17));b=rng.random(17)
        reference=[];draw=np.random.default_rng(0)
        for _ in range(57):
            idx=draw.integers(0,17,size=17)
            seeds=draw.choice([0,1,2,3,4],size=5,replace=True)
            reference.append(c[seeds][:,idx].mean()-b[idx].mean())
        got=m.ret_interval(c,b,57)
        np.testing.assert_allclose([got['lo'],got['hi']],np.percentile(reference,[2.5,97.5]),atol=1e-15)

    def test_ami_interval_transcript(self):
        y=np.array([0,0,1,1,-1,2,2])
        base=np.array([[0,0,1,1,2,2,2],[1,1,0,0,2,1,2],[0,1,0,1,2,2,2]])
        cand=np.stack([np.roll(base,s,axis=1) for s in range(5)])
        rng=np.random.default_rng(0);boot=[]
        for _ in range(19):
            idx=rng.integers(0,len(y),size=len(y));ss=rng.choice(5,size=5,replace=True)
            w=np.bincount(idx,minlength=len(y))
            boot.append(m.ami(cand,y,w,ss)-m.ami(base,y,w))
        got=m.ami_interval(cand,base,y,19,1)
        np.testing.assert_allclose([got['lo'],got['hi']],np.percentile(boot,[2.5,97.5]),atol=1e-15)

    def test_self_exclusion_and_stable_ties(self):
        x=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],np.float32)
        rank=m.ranking(x,x,np.arange(4))
        self.assertEqual(rank.tolist(),[[1,2,3],[0,2,3],[3,0,1],[2,0,1]])
        ap,_=m.aps(x,np.array([0,0,1,1]))
        np.testing.assert_array_equal(ap[0],np.ones(4))

    def test_missing_labels_excluded_before_gallery(self):
        x=m.unit(np.array([[1,0],[1,1],[0,1],[1,0]],np.float32))
        ap,rows=m.aps(x,np.array([0,-1,1,0]))
        self.assertEqual(rows.tolist(),[0,2,3])
        self.assertTrue(np.isnan(ap[0][1]))
        self.assertEqual(ap[0][[0,2]].tolist(),[1.,1.])

    def test_selection_guard_and_tie(self):
        rows={'none+k0':{'emotion':.4},'center+k0':{'emotion':.50000},
              'none+k1':{'emotion':.50001},'whiten64+k0':{'emotion':.1}}
        got=m.select(rows,{'emotion':.1})
        self.assertEqual(got['pick'],'center+k0')
        self.assertFalse(got['table']['whiten64+k0']['kept'])
        rows['none+k0']['emotion']=.5
        self.assertEqual(m.select(rows,{'emotion':.1})['pick'],'none+k0')

    def test_outcome_strict_boundaries(self):
        self.assertEqual(m.outcome(0.,.01),'LEVEL')
        self.assertEqual(m.outcome(-.01,.01),'BELOW')
        self.assertEqual(m.outcome(1e-12,.01),'WIN')

    def test_ladder_inconclusive_can_raise(self):
        rows=[dict(id=f'c{i}/{side}',outcome='WIN') for i in range(6) for side in ('p1','q1')]
        self.assertEqual(m.ladder(rows,6,5)['level'],3)
        rows[0]['outcome']='INCONCLUSIVE'
        got=m.ladder(rows,6,5)
        self.assertEqual(got['level'],1)
        self.assertTrue(got['inconclusive_above'])
        rows[0]['outcome']='BELOW'
        self.assertFalse(m.ladder(rows,6,5)['inconclusive_above'])

    def test_join_rejects_duplicate_identity(self):
        self.assertEqual(m.join(['b','a','c'],['c','b']).tolist(),[2,0])
        with self.assertRaises(AssertionError):m.join(['a','a'],['a'])
        with self.assertRaises(KeyError):m.join(['a'],['b'])

    def test_map_normalizes_in_float64_then_casts(self):
        rng=np.random.default_rng(2)
        x=m.unit(rng.normal(size=(21,4)).astype(np.float32))
        mu,mat=m.mapping(x,'center')
        expected=x.astype(np.float64)-x.astype(np.float64).mean(0)
        expected=(expected/np.linalg.norm(expected,axis=1,keepdims=True)).astype(np.float32)
        np.testing.assert_array_equal(m.mapped(x,(mu,mat)),expected)

    def test_none_readout_still_normalizes_in_float64(self):
        rng=np.random.default_rng(123)
        x=m.unit(rng.normal(size=(31,6)).astype(np.float32))
        y=np.arange(31)%3
        args=SimpleNamespace(query_limit=0,batch_size=8,device='cpu')
        got=m.image_ap(x,x,y,'none+k1',args)
        z=m.unit(x,np.float64).astype(np.float32)
        expected=m.aps(z,y,(1,),batch=8)[0][1]
        np.testing.assert_array_equal(got,expected)

    def test_controller_guard_before_any_data_read(self):
        result=subprocess.run([sys.executable,str(Path(m.__file__)),'held'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('controller-only',result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_held_missing_pick_fixture_has_all_cell_ids_and_no_reads(self):
        class FakeIO:
            def json(self,path):return {'mode':'validation'}
            def file(self,path):return 'fixture-digest'
            def envelope(self,**content):return content
        work=m.Work.__new__(m.Work)
        work.io=FakeIO()
        work.args=SimpleNamespace(picks_committed=True,val_sha256='fixture-digest',
            val_json='fixture',replay_json=[],piece='all',comparators=('p1','q1'),
            seeds=m.SEEDS,aspects=m.ASPECTS,query_limit=0,quick=False,metric='all',
            clu_draws=2000,ret_draws=10000)
        result=work.held()
        self.assertEqual(len(result['mmart']['cells']),12)
        self.assertEqual(len(result['photo']['cells']),8)
        self.assertEqual(result['photo']['sc6']['level'],0)
        self.assertTrue(result['mmart']['sc5']['inconclusive_above'])
        self.assertTrue(all(c['value'] is None for piece in ('mmart','photo') for c in result[piece]['cells']))

    def test_replay_fragments_require_distinct_complete_evidence(self):
        def doc(rows):
            return {'mode':'frozen_replay','scope':{'full_protocol':True,'aspects':list(m.ASPECTS)},
                    'inputs':{str(Path(m.__file__).resolve()):m.sha(m.__file__)},'replays':rows}
        def rows(family,n):
            return [{'cell_type':family,'target_record':'synthetic-fixture','key':str(i),'passed':True} for i in range(n)]
        core=rows('affection/map',10)
        rebuild=rows('affection/map/rebuild',2)
        val=rows('affection/validation/map',25)+rows('affection/validation/pick',1)
        self.assertTrue(m.replay_pass([doc(core),doc(rebuild[:1]),doc(rebuild[1:]),doc(val)],'affection','map'))
        self.assertFalse(m.replay_pass([doc(core),doc(rebuild[:1]),doc(val)],'affection','map'))
        core[0]['passed']=False
        self.assertFalse(m.replay_pass([doc(core),doc(rebuild),doc(val)],'affection','map'))

    def test_replay_proof_fails_closed(self):
        self.assertFalse(m.replay_pass([], 'mmart','map'))
        self.assertFalse(m.replay_pass([{'mode':'held'}],'affection','ami'))


if __name__=='__main__':unittest.main()
