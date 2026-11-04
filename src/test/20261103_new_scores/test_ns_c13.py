"""Quick mathematical and decision tests; no saved-input work or round outputs."""
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score

import ns_c13 as c


def brute_acc(y,p):
    classes=np.unique(y);clusters=np.unique(p)
    n=max(len(classes),len(clusters));table=np.zeros((n,n),int)
    for i,a in enumerate(classes):
        for j,b in enumerate(clusters):table[i,j]=np.sum((y==a)&(p==b))
    return max(sum(table[i,j] for i,j in enumerate(perm)) for perm in itertools.permutations(range(n)))/len(y)


@pytest.mark.parametrize('seed',range(6))
def test_metrics_repeated_rows(seed):
    rng=np.random.default_rng(seed)
    y=rng.integers(-1,4,27);p=rng.integers(0,4,27)
    rows=rng.integers(0,27,27);w=np.bincount(rows,minlength=27)
    yr,pr=y[rows],p[rows];mask=yr>=0;yr,pr=yr[mask],pr[mask]
    got=c.metrics(c.contingency(y,p,w,shape=(6,7)))
    expected=[adjusted_mutual_info_score(yr,pr),normalized_mutual_info_score(yr,pr,average_method='arithmetic'),adjusted_rand_score(yr,pr),brute_acc(yr,pr)]
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-12)


@pytest.mark.parametrize('y,p', [([0],[0]),([0,0],[2,2]),([0,1],[0,1]),([0,1,2],[0,0,0]),([0,0,1,1],[0,1,0,1])])
def test_metric_degeneracies(y,p):
    got=c.metrics(c.contingency(y,p))
    np.testing.assert_allclose(got[:3],[adjusted_mutual_info_score(y,p),normalized_mutual_info_score(y,p),adjusted_rand_score(y,p)],atol=1e-12,rtol=0)
    assert got[3]==brute_acc(np.asarray(y),np.asarray(p))


def test_weighted_mask_and_empty_cluster():
    y=np.array([0,-1,1,0]);p=np.array([0,3,2,1]);w=np.array([3,8,2,0])
    table=c.contingency(y,p,w,shape=(2,4))
    np.testing.assert_array_equal(table,[[3,0,0,0],[0,0,2,0]])
    assert c.metrics(table)[3]==1.
    with pytest.raises(AssertionError):c.contingency(y,p,np.array([1,-1,2,3]))


def test_draw_order():
    rng=np.random.default_rng(0)
    for rows,s,t in c.draws(13,6,True):
        np.testing.assert_array_equal(rows,rng.integers(0,13,size=13))
        np.testing.assert_array_equal(s,rng.choice(5,size=5,replace=True))
        np.testing.assert_array_equal(t,rng.choice(5,size=5,replace=True))
    rng=np.random.default_rng(0)
    for rows,s,t in c.draws(13,6,False):
        np.testing.assert_array_equal(rows,rng.integers(0,13,size=13))
        np.testing.assert_array_equal(s,rng.choice(5,size=5,replace=True));assert t is None


def test_joint_bootstrap_against_expanded_rows():
    rng=np.random.default_rng(7);y=np.array([0,1,-1,2,1,0,0,2,2])
    a=rng.integers(0,3,(5,3,9));b=rng.integers(0,3,(5,3,9))
    got=c.joint_clustering(a,b,y,7,1);boot=[]
    def arm(p,rows,seeds):
        return np.mean([np.mean([c.metrics(c.contingency(y[rows],state[rows])) for state in p[s]],axis=0) for s in seeds],axis=0)
    for rows,s,t in c.draws(9,7,True):boot.append(arm(a,rows,s)-arm(b,rows,t))
    bounds=np.percentile(boot,[2.5,97.5],axis=0)
    for i,m in enumerate(c.METRICS):np.testing.assert_allclose([got[m]['lo'],got[m]['hi']],bounds[:,i],atol=1e-12,rtol=0)


def test_apart_retrieval_draws():
    rng=np.random.default_rng(1);a=rng.random((5,23));b=rng.random((5,23));boot=[]
    for rows,s,t in c.draws(23,19,True):boot.append(a[s][:,rows].mean()-b[t][:,rows].mean())
    got=c.paired_retrieval(a,b,19,True)
    np.testing.assert_allclose([got['lo'],got['hi']],np.percentile(boot,[2.5,97.5]),atol=1e-14,rtol=0)
    got=c.paired_retrieval(a,b[0],19)
    ref=c.rt_stats.paired(a,b[0],19)
    assert [got['lo'],got['hi']]==ref['interval']


def test_untold_order_dtype_keys():
    ks=['b','a'];heads=[(ks,np.array([[i,2],[3,i]],np.float16)) for i in (1,2,4)]
    have,x=c.untold_join(heads)
    assert have==ks and x.dtype==np.float32
    expected=np.concatenate([c.unit(h) for _,h in heads],1)/np.float32(np.sqrt(3))
    np.testing.assert_array_equal(x,expected)
    np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-7)
    heads[1]=(ks[::-1],heads[1][1])
    with pytest.raises(AssertionError,match='key lists differ'):c.untold_join(heads)


@pytest.mark.parametrize('lo,r,word',[(.01,.1,'WIN'),(0,.1,'LEVEL'),(-.1,.1,'BELOW'),(-.099,.1,'LEVEL'),(0,0,'BELOW')])
def test_words(lo,r,word):assert c.outcome(lo,r)==word


def good_status():
    return dict(gates={**dict.fromkeys(('i_a','i_b','i_d','ii_scorer','ii_crl','ii_percept','vi'),'ok'),
        'i_c':dict.fromkeys(c.SETS[2:],'ok'),'iii_metric':dict.fromkeys(c.METRICS,'ok'),
        'iii_count':{d+'/'+a:'ok' for d in c.SETS for a in c.K[d]},'iv':dict.fromkeys(c.ENCODERS,'ok'),'v':dict.fromkeys(c.ENCODERS,'ok')},
        stage1={'ranges':{c.range_key(x):'ok' for x in c.claim_cells() if x['kind']=='class_count'},
                'picks':{'/'.join(x):'ok' for x in c.PICK_UNITS}})


def cell(sc='SC3',d='mmart',a='emotion'):
    return {**next(x for x in c.claim_cells() if x['sc']==sc and x['set']==d and x['aspect']==a), 'lo':-.005,'r':.01}


def test_bar_and_teacher_sentences():
    x=cell();r=c.decision(x)
    assert x['bar']=='WIN' and r['outcome']=='LEVEL' and not r['bar_met'] and r['meets_bar']
    assert r['wording']=='level with or better than'
    assert c.decision({**x,'lo':.01})['wording']=='better than'
    x=cell(d='artelingo')
    assert x['bar']=='LEVEL_OR_WIN' and c.decision({**x,'lo':.01})['wording']=='level with or better than'
    x=cell('SC1')
    assert not c.decision(x)['meets_bar'] and c.decision(x)['wording'] is None


def test_void_scopes_and_ranges():
    st=good_status();x=cell();assert c.decision(x,st)['status']=='decided'
    st['gates']['i_a']='failed'
    assert c.decision(x,st)['status']=='inconclusive'
    assert c.decision(cell('SC4','emoset'),st)['status']=='decided'
    st=good_status();st['gates']['iv']['qwen']='failed'
    assert not c.voids(cell('SC8','artelingo'),st)
    assert c.voids(cell('SC8','mmart'),st)
    assert c.voids(cell('SC2','mmart'),st)
    assert not c.voids(cell('SC2','artelingo'),st)
    st=good_status();st['stage1']['ranges'][c.range_key(x)]='missing'
    assert c.decision(x,st)['status']=='inconclusive'
    assert c.decision({**x,'lo':.001},st)['status']=='decided'
    assert c.decision({**x,'lo':-.02},st)['status']=='inconclusive'
    y={**cell('SC1'),'lo':-.02}
    st['stage1']['ranges'][c.range_key(y)]='missing'
    assert c.decision(y,st)['status']=='decided'


def test_new_pick_gate_and_missing_key():
    st=good_status();x=cell('SC7');st['gates']['vi']='failed'
    assert c.voids(x,st)
    crl=next(x for x in c.claim_cells() if x['sc']=='SC6' and x['set']=='artelingo')
    assert not c.voids(crl,st)
    st=good_status();st['stage1']['picks']={}
    assert c.voids(x,st)


def test_rule_coverage_and_section_estimates():
    assert len(c.claim_cells())==412
    assert len([s for s in c.SECTIONS if s.startswith('cells:')])==21
    assert len(c.PICK_UNITS)==16
    assert all(c.estimate(s)>0 for s in c.SECTIONS)
    assert sum(map(c.estimate,c.SECTIONS))<600
    assert not c.full_scope({})


def test_tolerance_and_exact_categories():
    r=c.Comparison();r.add('a/x',1.,1.+.9e-6,True);r.add('a/y',1.,1.+1.1e-6,True)
    r.add('a/bool',True,1);r.add('a/word','WIN','win')
    assert [x['pass'] for x in r.items]==[True,False,False,False]
    assert r.report()['failed']==3


@pytest.mark.parametrize('scope,failed,skipped,draws,content',[
    ('subset',0,[],[('clu',2000)],b'{}'),('full',1,[],[('clu',2000)],b'{}'),
    ('full',0,[],[('clu',40)],b'{}'),('full',0,[],[('ret',100)],b'{}'),
    ('full',0,['missing status'],[('clu',2000)],b'{}'),('full',0,[],[('clu',2000)],None)])
def test_refuse_record(scope,failed,skipped,draws,content):
    with patch.object(Path,'mkdir') as mkdir:
        with pytest.raises(ValueError,match='refusing --write-record'):
            c.certify(dict(scope=scope,failed=failed,skipped=skipped),'/forbidden',content,draws)
        mkdir.assert_not_called()


def test_missing_dependency_stops_without_computation():
    with patch.object(Path,'exists',return_value=False):
        with pytest.raises(FileNotFoundError,match='missing prerequisite'):
            c.Work(c.SMOKE_OUT).dependency('percept_val')


def test_output_roots():
    assert c.output_path(c.SMOKE_OUT/'cache/a.npz').is_relative_to(c.SMOKE_OUT)
    with pytest.raises(ValueError):c.output_path('/tmp/forbidden')


def test_independent_pick_score_guard_and_tie():
    ros=c.rt_readout.menu(40);rng=np.random.default_rng(8)
    rows={ro:dict(zip(c.ASPECTS,rng.uniform(.25,.35,3))) for ro in ros}
    ranges=dict(zip(c.ASPECTS,(.03,.02,.04)))
    pick,scores,guards=c.select_pick(rows,ranges)
    assert pick==c.ac_select.pick_readout(rows,ranges)
    assert scores==c.ac_select.readout_scores(rows,ranges)
    assert guards==c.ac_select.readout_guard(rows,ranges)
    r0=c.rt_readout.R0;other=c.rt_readout.ReadOut('center',None,0)
    rows={r0:dict.fromkeys(c.ASPECTS,.3),other:dict.fromkeys(c.ASPECTS,.300009)}
    assert c.select_pick(rows,dict.fromkeys(c.ASPECTS,.01))[0]==r0
    rows[other]=dict.fromkeys(c.ASPECTS,.300011)
    assert c.select_pick(rows,dict.fromkeys(c.ASPECTS,.01))[0]==other


def test_compare_subset_lists_absent_cells_and_refuses_record():
    section='cells:clu:emoset:qwen'
    cells=[{**x,'cand':.4,'comp':.3,'d':.1,'lo':.01,'hi':.2,'r':.03,'n_draws':40}
           for x in c.claim_cells() if (x['set'],x['task'])==('emoset','clu')]
    doc=dict(section=section,complete=True,source_sha256=c.source_identity(),draws_clu=40,draws_ret=100,result=cells)
    root=c.SMOKE_OUT/'unit_round';out=c.SMOKE_OUT/'unit_compare'
    with patch.object(Path,'glob',return_value=[out/'derived'/f'{section}.json']), \
         patch.object(Path,'exists',return_value=False), \
         patch.object(c,'read_json',return_value=doc),patch.object(c,'write_json') as writer:
        report=c.compare(root,root/'records',out)
        assert report['failed']==0 and report['scope']=='subset'
        assert len(report['missing_round_cells'])==8
        assert any('outcomes' in s for s in report['skipped'])
        assert any('status' in s for s in report['skipped'])
        assert writer.call_count==1
        with pytest.raises(ValueError,match='scope is a subset'):
            c.compare(root,root/'records',out,write_record=True)


@pytest.mark.parametrize('gate,sc,d,affected',[
    ('i_b','SC2','artelingo',True),('i_b','SC1','artelingo',False),
    ('i_d','SC1','mmart',True),('ii_scorer','SC7','artelingo',True),
    ('ii_scorer','SC1','artelingo',False),('ii_crl','SC5','mmart',True),
    ('ii_crl','SC7','mmart',False),('ii_percept','SC8','mmart',True),
    ('ii_percept','SC7','mmart',False)])
def test_gate_table(gate,sc,d,affected):
    st=good_status();st['gates'][gate]='failed'
    x=cell(sc,d)
    assert bool(c.voids(x,st))==affected


@pytest.mark.parametrize('k',(8,20,27))
def test_exact_emi_at_round_populations(k):
    from sklearn.metrics.cluster import expected_mutual_information
    rng=np.random.default_rng(k);prob=rng.random(k*k);prob/=prob.sum()
    C=rng.multinomial(16384,prob).reshape(k,k)
    assert abs(c.expected_mi(C,16384)-expected_mutual_information(C,16384))<1e-12


@pytest.mark.parametrize('dataset,threads',[('artelingo',8),('mmart',8),('affection',1),('emoset',1)])
def test_partition_settings_preserve_stored_rows(dataset,threads):
    from types import SimpleNamespace
    x=np.array([[1.0001,0],[0,.9999]],dtype=np.float32);seen=[]
    def factory(**kwargs):
        def fit(got):
            assert got is x
            seen.append(kwargs)
            return SimpleNamespace(labels_=np.array([0,1]))
        return SimpleNamespace(fit=fit)
    with patch.object(c,'MiniBatchKMeans',side_effect=factory),patch.object(c,'threadpool_limits') as limits:
        p=c.partitions(x,2,dataset)
        limits.assert_called_once_with(threads)
    assert p.shape==(3,2) and p.dtype==np.int32
    assert [v['random_state'] for v in seen]==[0,1,2]
    assert all(v['n_init']==3 and v['batch_size']==4096 and v['n_clusters']==2 for v in seen)


def test_full_scope_requires_every_unit_cell_table_and_protocol_draws():
    source=c.source_identity()
    docs={s:dict(section=s,complete=True,source_sha256=source,draws_clu=2000,draws_ret=10000,result={}) for s in c.SECTIONS}
    docs['counts']['result']={d:{split:c.K[d] for split in ('val',c.HELD[d])} for d in c.SETS}
    docs['percept_val']['result']={e:{str(s):{} for s in c.SEEDS} for e in c.ENCODERS}
    for d,e in c.PAIRS:
        arm='cstar' if d in c.SETS[:2] else 'photo'
        docs[f'ranges:{d}:{e}']['result']={f'{arm}/{d}/{e}/{a}/{m}':{'per_seed':dict.fromkeys(map(str,c.SEEDS),.1)} for a in c.K[d] for m in c.METRICS}
    for arm,d,e in c.PICK_UNITS:
        width=6144 if arm=='untold' else 40 if arm=='matched' else 50
        docs[f'pick:{arm}:{d}:{e}']['result']={'table':{c.rt_readout.name(ro):{} for ro in c.rt_readout.menu(width)}}
    for section,doc in docs.items():
        if section.startswith('cells:'):
            _,task,d,e=section.split(':')
            doc['result']=[{**x,'n_draws':2000 if task=='clu' else 10000} for x in c.claim_cells() if (x['task'],x['set'],x['encoder'])==(task,d,e)]
    assert c.full_scope(docs)
    for name,doc in docs.items():
        if not name.startswith('cells:'):doc['draws_clu'],doc['draws_ret']=40,100
    assert c.full_scope(docs)  # point-only prerequisites have no bootstrap draw count
    section='cells:clu:emoset:qwen';docs[section]['result'].pop()
    assert not c.full_scope(docs)
    docs[section]['result']=[{**x,'n_draws':2000} for x in c.claim_cells() if x['set']=='emoset']
    docs[section]['draws_clu']=40
    assert not c.full_scope(docs)
    docs[section]['draws_clu']=2000
    docs['pick:untold:artelingo:qwen']['result']['table'].pop('none+k0')
    assert not c.full_scope(docs)


def test_nested_void_scopes():
    st=good_status();st['gates']['i_c']['affection']='missing'
    assert c.voids(cell('SC4','affection'),st)
    assert not c.voids(cell('SC4','emoset'),st)
    st=good_status();st['gates']['iii_metric']['ami']='failed'
    x=cell('SC1','artelingo')
    assert c.voids(x,st)
    assert not c.voids({**x,'metric':'ari'},st)
    st=good_status();st['gates']['iii_count']['artelingo/emotion']='failed'
    assert c.voids(x,st)
    fixed=next(x for x in c.claim_cells() if x['set']=='artelingo' and x['kind']=='fixed20')
    assert not c.voids(fixed,st)
    st=good_status();st['gates']['v']['qwen']='failed'
    assert c.voids(cell('SC7','artelingo'),st)
    assert not c.voids(cell('SC8','artelingo'),st)


def test_record_hashes_exact_outcomes_bytes():
    import hashlib
    content=b'{"rows": []}\n'
    report=dict(scope='full',failed=0,skipped=[],items_compared=8123,max_abs=2e-14)
    with patch.object(Path,'mkdir'),patch.object(Path,'write_text') as write,patch.object(Path,'replace') as replace:
        c.certify(report,Path('/virtual'),content,[('clu',2000),('ret',10000)])
        written=json.loads(write.call_args.args[0])
        assert written=={'pass':True,'outcomes_sha256':hashlib.sha256(content).hexdigest(),'items':8123,'max_abs':2e-14}
        assert replace.call_args.args[0]==Path('/virtual/c13.json')


def test_compare_checks_present_numbers_with_tolerance():
    section='cells:clu:emoset:qwen';out=c.SMOKE_OUT/'unit_compare';root=c.SMOKE_OUT/'unit_round'
    cells=[{**x,'cand':.4,'comp':.3,'d':.1,'lo':.01,'hi':.2,'r':.03,'n_draws':40,'n_rows':9,'n_labelled':9}
           for x in c.claim_cells() if (x['set'],x['task'])==('emoset','clu')]
    doc=dict(section=section,complete=True,source_sha256=c.source_identity(),draws_clu=40,draws_ret=100,result=cells)
    refs=[dict(x) for x in cells];refs[0]['cand']+=2e-6;refs[1]['cand']+=.5e-6
    target=root/'stage2/cells/clu__emoset__qwen.json'
    with patch.object(Path,'glob',return_value=[out/'derived'/f'{section}.json']), \
         patch.object(Path,'exists',lambda p:p==target), \
         patch.object(c,'read_json',side_effect=lambda p:refs if p==target else doc), \
         patch.object(c,'write_json'):
        report=c.compare(root,root/'records',out)
    assert report['failed']==1
    assert [x['name'] for x in report['items'] if not x['pass']][0].endswith('/cand')
    assert not report['missing_round_cells']
