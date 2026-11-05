"""Quick independent r18 C13 tests; all temporary artifacts stay under c13_smoke."""
import hashlib
import json
from pathlib import Path
import tempfile
import importlib.util
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
import om_c13 as c


@pytest.fixture
def scratch():
    root=c.SMOKE_OUT/'unit'
    root.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def put(path, doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))


def test_official_clay_reference():
    assert c.sha(c.PARITY/'inputs.npz')==c.PARITY_SHA256['inputs']
    assert c.sha(c.PARITY/'clay_ref.npz')==c.PARITY_SHA256['clay']
    with np.load(c.PARITY/'inputs.npz') as z:x,t=z['clip'],z['text']
    with np.load(c.PARITY/'clay_ref.npz') as z:ref={k:z[k] for k in z.files}
    with threadpool_limits(2):
        p,mu=c.clay_subspace(t,49);v=c.clay_condition(x,p,mu)
    for a,b in [(p,ref['P']),(mu,ref['mu']),(v,ref['query']),(v,ref['gallery'])]:
        assert a.dtype==np.float32
        assert np.max(np.abs(a-b))<1e-5


def test_reflections_align_mean_and_identity():
    rng=np.random.default_rng(9)
    a,b=c.unit64(rng.normal(size=(2,15)))
    assert np.max(np.abs(c.align(a,a,b)-b))<1e-14
    x=rng.normal(size=(4,15))
    assert np.array_equal(c.reflection(x,np.zeros(15)),x)
    assert np.array_equal(c.reflection(x,np.full(15,1e-8)),x)
    assert np.max(np.abs(c.align(x,a,a)-x))<1e-14


def test_clay_split_mean_and_float_path():
    rng=np.random.default_rng(1)
    x=c.unit64(rng.normal(size=(20,8))).astype(np.float32)
    t=c.unit64(rng.normal(size=(7,8))).astype(np.float32)
    p,mu=c.clay_subspace(t)
    v=c.clay_condition(x,p,mu)
    assert np.max(np.abs(np.linalg.norm(v,axis=1)-1))<2e-7
    assert not np.allclose(v[:10],c.clay_condition(x[:10],p,mu))
    with pytest.raises(ValueError):c.clay_subspace(t.astype(np.float64))


def store(root, rows, vals):
    root.mkdir(parents=True,exist_ok=True)
    put(root/'keys.json',rows)
    np.save(root/'emb.npy',np.asarray(vals,dtype=np.float32))


def test_tpips_join_and_mmart_val(scratch):
    roots={'artelingo':scratch/'ae','mmart':scratch/'mm'}
    store(roots['artelingo']/'style',['b','a'],[[2,0],[0,3]])
    store(roots['mmart']/'style',['m:a','m:b'],[[4,0],[0,5]])
    assert np.array_equal(c.tpips_join('artelingo','held','style',['a','b'],roots),[[0,3],[2,0]])
    assert np.array_equal(c.tpips_join('mmart','test','style',['m:b','m:a'],roots),[[0,5],[4,0]])
    assert np.array_equal(c.tpips_join('mmart','val','style',['m:a','m:b'],roots,lambda k:k[2:]),[[0,3],[2,0]])
    with pytest.raises(ValueError,match='missing'):c.tpips_join('artelingo','val','style',['z'],roots)
    with pytest.raises(ValueError,match='duplicate'):c.tpips_join('artelingo','val','style',['a','a'],roots)
    put(roots['artelingo']/'style/keys.json',['a','a'])
    with pytest.raises(ValueError,match='duplicate'):c.tpips_join('artelingo','val','style',['a'],roots)


def make_parts(root,arm='plain'):
    paths=[]
    for i in range(3 if arm=='plain' else 5):
        name=f'k9__start{i}' if arm=='plain' else f'emotion__s{i}__k9'
        p=root/'parts/turtle/mmart/qwen'/arm/(name+'.npz')
        p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p,keys=np.asarray(['a','b','c']),labels=np.asarray([i,1,0],dtype=np.int32))
        side=dict(method='turtle',set='mmart',split='test',enc='qwen',arm=arm,
                  aspect=None if arm=='plain' else 'emotion',K=9,start=i if arm=='plain' else 0,
                  seed=None if arm=='plain' else i,n=3,sha256=c.sha(p))
        put(p.with_suffix('.json'),side);paths.append(p)
    return paths


def test_turtle_shapes_and_repeat(scratch):
    make_parts(scratch);make_parts(scratch,'cstar')
    p=c.turtle_parts('mmart','qwen','plain',9,['a','b','c'],root=scratch)
    q=c.turtle_parts('mmart','qwen','cstar',9,['a','b','c'],'emotion',root=scratch)
    assert p.shape==(3,3) and q.shape==(5,1,3)
    rep=c.repeat_turtle(q)
    assert rep.shape==(5,3,3) and np.array_equal(rep[:,0],rep[:,2])
    y=np.asarray([0,1,2])
    assert np.array_equal(c.ns.arm_metrics(q,y),c.ns.arm_metrics(rep,y))
    with pytest.raises(ValueError):c.repeat_turtle(rep)


@pytest.mark.parametrize('fault',['keys','sha','metadata','labels','dtype','missing'])
def test_turtle_refuses(scratch,fault):
    p=make_parts(scratch)[0];side=c.read_json(p.with_suffix('.json'))
    if fault=='keys':
        with pytest.raises(ValueError,match='keys'):c.turtle_parts('mmart','qwen','plain',9,['b','a','c'],root=scratch)
        return
    if fault=='sha':side['sha256']='wrong'
    elif fault=='metadata':side['start']=4
    elif fault in ('labels','dtype'):
        np.savez(p,keys=np.asarray(['a','b','c']),labels=np.asarray([0,1,9],dtype=np.int32) if fault=='labels' else np.asarray([0,1,2],dtype=np.int64))
        side['sha256']=c.sha(p)
    elif fault=='missing':
        with pytest.raises(FileNotFoundError):c.turtle_parts('mmart','qwen','cstar',9,['a','b','c'],'emotion',root=scratch)
        return
    put(p.with_suffix('.json'),side)
    with pytest.raises(ValueError):c.turtle_parts('mmart','qwen','plain',9,['a','b','c'],root=scratch)


def test_cell_list_and_range_sources():
    cells=c.cell_specs()
    assert len(cells)==len({x['id'] for x in cells})==337
    assert {sc:sum(x['sc']==sc for x in cells) for sc in c.SC}==dict(SC1=29,SC2=6,SC3=58,SC4=12,SC5=116,SC6=116)
    assert len(c.UNITS)==28
    for x in cells:
        src=c.range_source(x);d,e,a,m=(x[k] for k in ('set','encoder','aspect','metric'))
        if m=='map':assert src==('r11/map' if (d,e)==('artelingo','qwen') else 'r15/ret_ranges' if d=='mmart' else 'r14/artret_ranges')
        elif m=='ami' and x['k']==20:assert src==('r15/clu_ranges' if d=='mmart' else 'computed/qwen_ami20' if e=='qwen' else 'r14/artclu')
        else:assert src==f'r17/cstar/{d}/{e}/{a}/{m}'
        assert c.range_key(x).startswith(f'{d}/{e}/{a}/')
    rows=[x for x in cells if x['set']=='mmart' and x['aspect']=='style' and x['sc']=='SC1']
    assert len(rows)==4 and {x['k'] for x in rows}=={20}
    assert all(c.class_count_cell(x) for x in rows)


@pytest.mark.parametrize('task,margin',[('clu',.001),('ret',.1)])
def test_textset_boundary(task,margin):
    assert c.choose_textset({'A':0},task)=='A'
    for offset,expect in [(-1e-10,'A'),(0,'A'),(c.rt.SLACK/2,'A'),(2*c.rt.SLACK,'B')]:
        assert c.choose_textset({'A':0.,'B':margin+offset},task)==expect


@pytest.mark.parametrize('lo,r,expect',[(.0001,None,'WIN'),(0,None,'inconclusive'),(-1,None,'inconclusive'),
                                      (0,.1,'LEVEL'),(-.1,.1,'BELOW'),(-.099,.1,'LEVEL'),(0,0,'BELOW')])
def test_label_words(lo,r,expect):
    assert c.label(lo,r)==expect


def statuses():
    status={'gates':{k:'ok' for k in ('i','ii','iii_a','iii_b','iii_c','iv')},'stage1':{}}
    previous={'gates':dict(i_a='ok',i_d='ok',ii_scorer='ok',vi='ok',
                          iii_metric={m:'ok' for m in c.METRICS},iii_count={f'{d}/{a}':'ok' for d in c.SETS for a in c.ASPECTS}),
              'stage1':{'ranges':{}}}
    for row in c.cell_specs():
        status['stage1'][c.range_key(row)]='ok'
        previous['stage1']['ranges']['/'.join(('cstar',row['set'],row['encoder'],row['aspect'],row['metric']))]='ok'
    for d in c.SETS:
        for key in (f'tpips/{d}',f'clay/{d}',f'clay_textset/ret/{d}',f'clay_textset/clu/{d}'):status['stage1'][key]='ok'
    return status,previous


def row_for(sc='SC1',d='mmart',metric='ami',k=9):
    row=next(x for x in c.cell_specs() if x['sc']==sc and x['set']==d and x['metric']==metric and (metric=='map' or x['k']==k))
    return {**row,'lo':-.01,'r':.1}


@pytest.mark.parametrize('gate,voided',[('i',{'SC3','SC4'}),('ii',{'SC5','SC6'}),
    ('iii_a',{'SC1','SC3','SC5','SC6'}),('iii_b',{'SC5','SC6'}),('iii_c',{'SC2','SC4'}),('iv',{'SC1','SC2'})])
def test_void_gate_table(gate,voided):
    status,previous=statuses();status['gates'][gate]='failed'
    for sc in c.SC:
        row=row_for(sc,metric='map' if sc in ('SC2','SC4') else 'ami')
        assert bool(c.decide(row,status,previous)['void'])==(sc in voided)


@pytest.mark.parametrize('key,sc',[('tpips/mmart','SC2'),('clay/mmart','SC4'),('clay_textset/ret/mmart','SC4'),('clay_textset/clu/mmart','SC3')])
def test_stage1_voids(key,sc):
    s,p=statuses();s['stage1'][key]='missing'
    for other in c.SC:
        row=row_for(other,metric='map' if other in ('SC2','SC4') else 'ami')
        assert bool(c.decide(row,s,p)['void'])==(other==sc)


def test_r17_voids_and_missing_range():
    s,p=statuses();row=row_for();assert c.decide(row,s,p)['label']=='LEVEL'
    s['stage1'][c.range_key(row)]='failed'
    assert c.decide(row,s,p)['label']=='inconclusive'
    assert c.decide({**row,'lo':.01},s,p)['label']=='WIN'
    s,p=statuses();p['stage1']['ranges']['cstar/mmart/qwen/emotion/ami']='failed'
    assert c.decide(row,s,p)['label']=='inconclusive'
    p['gates']['iii_metric']['ami']='failed'
    assert c.decide(row,s,p)['label'] is None
    s,p=statuses();p['gates']['i_d']='failed'
    assert c.decide(row,s,p)['label'] is None
    assert c.decide(row_for(k=20),s,p)['label']=='LEVEL'
    s,p=statuses();p['gates']['vi']='missing'
    assert c.decide(row_for('SC2',metric='map'),s,p)['label'] is None


def test_section_dependencies_and_absent(scratch):
    seen=set()
    for s in c.SECTIONS:
        assert set(c.dependencies(s))<=seen
        seen.add(s)
    with pytest.raises(FileNotFoundError,match='derive it first'):c.Work(scratch).dependency('counts')
    with pytest.raises(ValueError):c.output_path('/tmp/c13')


def test_compare_tolerance_and_categorical_exactness():
    result=c.Comparison()
    result.add('x/at',0.,1e-6,True)
    result.add('x/over',0.,1.000001e-6,True)
    result.add('x/types',1,1.0)
    result.add('x/null',None,None,True)
    result.add('x/nonfinite',float('nan'),0.,True)
    assert [x['pass'] for x in result.items]==[True,False,False,True,False]
    assert result.report()['failed']==3
    assert not c.full_scope({})


def subset_fixture(root):
    out=root/'derive';records=root/'records';roundroot=root/'round'
    s,p=statuses();old=root/'old-status.json';put(old,p)
    s['r17']={'path':str(old),'sha256':c.sha(old)};put(records/'status.json',s)
    row={**row_for(), 'cand':.4,'comp':.3,'d':.1,'hi':.2,'n_draws':40,'n_rows':3}
    section='cells:SC1:mmart:qwen'
    doc=dict(section=section,complete=True,source_sha256=c.source_identity(),draws_clu=40,draws_ret=100,result=[row])
    c.write_json(out/'derived'/f'{section}.json',doc)
    finished=c.finish(out,records)
    put(roundroot/'stage2/cells.json',{'cells':finished['cells']})
    put(records/'stage1_om_picks.json',{})
    put(records/'stage1_om_ranges.json',{})
    return out,records,roundroot


def test_subset_compare_scope_and_coverage(scratch,capsys):
    out,records,root=subset_fixture(scratch)
    report=c.compare(root,records,out)
    assert report['scope']=='subset' and report['failed']==0
    assert report['reference_draw_counts']=={'clu':[40],'ret':[]}
    assert any('full bootstrap' in s for s in report['skipped'])
    assert 'scope=subset' in capsys.readouterr().out
    refs=c.read_json(root/'stage2/cells.json')
    other={**refs['cells'][0],'id':'SC1/mmart/qwen/style/ami/20/tpips'}
    refs['cells'].append(other);put(root/'stage2/cells.json',refs)
    assert c.compare(root,records,out)['failed']>0


def test_compare_detects_stale_finish(scratch):
    out,records,root=subset_fixture(scratch)
    p=next((out/'derived').glob('*.json'));doc=c.read_json(p)
    doc['result'][0]['cand']+=.01;c.write_json(p,doc)
    report=c.compare(root,records,out)
    assert report['failed']>0


def test_b_fallback_excludes_entire_set(scratch,monkeypatch):
    a,b=scratch/'A',scratch/'B'
    a.mkdir();b.mkdir()
    for aspect in c.ASPECTS:
        np.save(a/f'{aspect}_text.npy',np.eye(3,dtype=np.float32))
        np.save(b/f'{aspect}_text.npy',np.eye(3,dtype=np.float32))
    rows={'aspects':{a:{'fallback':False} for a in c.ASPECTS}}
    put(b/'lists.json',rows)
    monkeypatch.setattr(c,'TEXT_A',a);monkeypatch.setattr(c,'TEXT_B',b)
    assert set(c.text_inputs())=={'A','B'}
    rows['aspects']['style']['fallback']=True;put(b/'lists.json',rows)
    assert set(c.text_inputs())=={'A'}


def test_array_cache_bound_to_rows_keys_and_source(scratch):
    w=c.Work(scratch)
    x=np.asarray([[1,2],[3,4]],dtype=np.float32)
    v,entry=w.stored('toy',[['a','b'],x],lambda:x+1)
    _,again=w.stored('toy',[['a','b'],x],lambda:pytest.fail('cache miss'))
    _,changed=w.stored('toy',[['b','a'],x],lambda:x+2)
    assert entry==again and entry['path']!=changed['path']
    assert np.array_equal(w.artifact(entry),v)
    with open(entry['path'],'ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'):w.artifact(entry)


@pytest.mark.parametrize('gate,sub,clu,ret,at20',[
    ('i_a',None,True,False,True),('i_d',None,True,False,False),
    ('iii_metric','ami',True,False,True),('iii_count','mmart/emotion',True,False,False),
    ('ii_scorer',None,False,True,False),('vi',None,False,True,False)])
def test_all_r17_gate_routes(gate,sub,clu,ret,at20):
    s,p=statuses()
    if sub is None:p['gates'][gate]='failed'
    else:p['gates'][gate][sub]='failed'
    assert bool(c.decide(row_for(),s,p)['void'])==clu
    assert bool(c.decide(row_for('SC2',metric='map'),s,p)['void'])==ret
    assert bool(c.decide(row_for(k=20),s,p)['void'])==at20


def test_turtle_repeat_preserves_joint_draws():
    rng=np.random.default_rng(12)
    p=rng.integers(0,3,size=(5,1,24),dtype=np.int32)
    b=rng.integers(0,3,size=(3,24),dtype=np.int32)
    y=np.tile(np.arange(3),8);y[-1]=-1
    one=c.ns.joint_clustering(p,b,y,5,workers=1)
    repeat=c.ns.joint_clustering(c.repeat_turtle(p),b,y,5,workers=1)
    for metric in c.METRICS:
        for field in ('cand','comp','d','lo','hi'):
            assert abs(one[metric][field]-repeat[metric][field])<1e-15


def full_documents():
    docs={s:dict(section=s,complete=True,source_sha256=c.source_identity(),draws_clu=2000,draws_ret=10000,result={}) for s in c.SECTIONS}
    docs['counts']['result']={d:{split:c.K[d] for split in ('val',c.HELD[d])} for d in c.SETS}
    for cell in c.cell_specs():
        key=c.range_key(cell);row=dict(r=0.,source=c.range_source(cell))
        if row['source']=='computed/qwen_ami20':row['per_seed']=[0.]*5
        docs['ranges']['result'][key]=row
    for sc,d,e in c.UNITS:
        docs[f'cells:{sc}:{d}:{e}']['result']=[{**x,'n_draws':2000 if x['task']=='clu' else 10000} for x in c.cell_specs() if (x['sc'],x['set'],x['encoder'])==(sc,d,e)]
    for d in c.SETS:
        for split in ('val',c.HELD[d]):
            docs[f'clay:{d}:{split}']['result']=dict(b_available=True,sets={ts:dict(arrays={a:{} for a in c.ASPECTS},ranks={a:rank for a in c.ASPECTS}) for ts,rank in [('A',49),('B',50)]})
        table=lambda width:dict(pick='none+k0',table={c.rt.name(r):{} for r in c.rt.menu(width)})
        docs[f'pick:tpips:{d}']['result']=table(4096)
        docs[f'pick:clay:{d}']['result']={**table(49),'text_set':'A','sets':{'A':table(49),'B':table(50)}}
        docs[f'textset:clu:{d}']['result']={'set':'A'}
        needed={f'{a}/{k}':{} for a in c.ASPECTS for k in {c.K[d][a],20}}
        for e in c.ENCODERS:docs[f'parts:cstar:{d}:{e}']['result']={'arrays':needed}
        for arm in ('clay','tpips'):docs[f'parts:{arm}:{d}']['result']={'arrays':needed}
    return docs


def test_full_scope_demands_all_cells_tables_and_draws():
    docs=full_documents();assert c.full_scope(docs)
    docs['cells:SC1:mmart:qwen']['draws_clu']=40
    assert not c.full_scope(docs)
    docs['cells:SC1:mmart:qwen']['draws_clu']=2000
    docs['pick:clay:mmart']['result']['sets']['B']['table'].pop('none+k0')
    assert not c.full_scope(docs)
    assert not c.full_scope({k:v for k,v in full_documents().items() if k!='counts'})


def turtle_test_array(d,e,arm,a=None,seed=None):
    offset=1+10*c.SETS.index(d)+2*c.ENCODERS.index(e)
    if arm=='cstar':offset+=20+5*c.ASPECTS.index(a)+seed
    x=np.arange(6,dtype=np.float32).reshape(3,2)+offset
    return (x/np.linalg.norm(x,axis=1,keepdims=True))[:,::-1]


def turtle_test_sidecars(root,count=412):
    fits=[]
    for d in c.SETS:
        for e in c.ENCODERS:
            for k in (5,8,9,10,16,20,27,40):
                for start in range(3):fits.append((d,e,'plain',None,None,f'k{k}__start{start}'))
            for a in c.ASPECTS:
                for k in sorted({c.K[d][a],20}):
                    for seed in range(5):fits.append((d,e,'cstar',a,seed,f'{a}__s{seed}__k{k}'))
    assert len(fits)==412
    if count==413:fits.append((*fits[0][:5],'extra'))
    for d,e,arm,a,seed,stem in fits[:count]:
        x=turtle_test_array(d,e,arm,a,seed)
        digest=hashlib.sha256((x.dtype.str+str(x.shape)).encode()+np.ascontiguousarray(x).tobytes()).hexdigest()
        put(root/'parts/turtle'/d/e/arm/(stem+'.json'),dict(aspect=a,seed=seed,input_sha256=digest))


def comparison_fixture(root,monkeypatch,sidecars=412,low_draw=False,missing_range=False):
    """Exercise finish and the entire compare function, with tiny synthetic inputs."""
    out,records,roundroot=root/'derive',root/'records',root/'round'
    docs=full_documents()
    for row in docs['ranges']['result'].values():
        row['r']=.1
        if 'per_seed' in row:row['per_seed']=[0.,.025,.05,.075,.1]
    for section,doc in docs.items():
        if section.startswith('cells:'):
            for row in doc['result']:
                row.update(cand=.4,comp=.3,d=.1,lo=-.01,hi=.2,r=.1,n_rows=3)
                if row['task']=='ret':row.update(candidate_readout='none+k0',comparator_readout='none+k0')
    if low_draw:
        doc=docs['cells:SC1:mmart:qwen'];doc['draws_clu']=40
        for row in doc['result']:row['n_draws']=40
    picks={}
    for d in c.SETS:
        for arm in ('tpips','clay'):
            row=docs[f'pick:{arm}:{d}']['result']
            tables={'tpips':row} if arm=='tpips' else row['sets']
            for ts,table in tables.items():
                table.update(unit={a:.1 for a in c.ASPECTS},n_val=4,unit_score=2.)
                table['table']={ro:dict(map={a:.2 for a in c.ASPECTS},score=0.,kept=True) for ro in table['table']}
                name=f'{arm}__{d}'+(f'__{ts}' if arm=='clay' else '')
                put(roundroot/'stage1/tables'/(name+'.json'),dict(table=table['table'],pick=table['pick'],ranges=table['unit'],n_val=4))
            if arm=='clay':
                row.update(**row['sets']['A'])
                picks[f'clay_textset/ret/{d}']=dict(set='A',scores={ts:dict(
                    pick=t['pick'],score=t['unit_score'],map=t['table'][t['pick']]['map'],
                    r0_map=t['table']['none+k0']['map'],rank=49 if ts=='A' else 50) for ts,t in tables.items()})
            picks[f'{arm}/{d}']=dict(pick=row['pick'],w='none',k=0,n_val=4,unit=row['unit'],scores={
                **row['table'][row['pick']],'r0_map':row['table']['none+k0']['map']})
            if arm=='clay':picks[f'{arm}/{d}'].update(text_set='A',unit_score=2.)
        cluster=docs[f'textset:clu:{d}']['result']
        cluster.update(scores={ts:.1 for ts in ('A','B')},per_aspect={ts:{a:.1 for a in c.ASPECTS} for ts in ('A','B')})
        picks[f'clay_textset/clu/{d}']=dict(set='A',scores={ts:dict(mean_ami20=.1,ami20={a:.1 for a in c.ASPECTS}) for ts in ('A','B')})
    for section,doc in docs.items():c.write_json(out/'derived'/f'{section}.json',doc)
    status,previous=statuses();old=root/'old-status.json';put(old,previous)
    if missing_range:status['stage1']['mmart/qwen/emotion/ami/9']='failed'
    status['r17']={'path':str(old),'sha256':c.sha(old)};put(records/'status.json',status)
    finished=c.finish(out,records)
    put(roundroot/'stage2/cells.json',[{**row,'r':row['r_used']} for row in finished['cells']])
    put(records/'stage1_om_picks.json',picks)
    put(records/'stage1_om_ranges.json',docs['ranges']['result'])
    turtle_test_sidecars(roundroot,sidecars)
    calls=[]
    def vector(self,d,split,e,arm,a='emotion',s=None,retrieval=False):
        assert split==c.HELD[d] and retrieval==(arm=='cstar')
        tag=(d,e,arm,a if arm=='cstar' else None,s if arm=='cstar' else None)
        calls.append(tag)
        return turtle_test_array(*tag)
    monkeypatch.setattr(c.Work,'vector',vector)
    return out,records,roundroot,old,calls


def test_compare_full_clean_pass_and_turtle_cache(scratch,monkeypatch):
    out,records,root,_,calls=comparison_fixture(scratch,monkeypatch)
    report=c.compare(root,records,out)
    assert report['failed']==0 and report['scope']=='full'
    assert report['parts']['turtle_inputs']['items']==413
    assert len(calls)==len(set(calls))==128


@pytest.mark.parametrize('fault,part',[
    ('label','cells:'),('void','cells:'),('readout','cells:'),('pick','picks/'),
    ('table','tables:'),('range','ranges/'),('per_seed','ranges/'),
    ('status','integrity/status_sha256'),('r17_status','integrity/r17_status_sha256'),
    ('turtle','turtle_inputs/')])
def test_compare_full_rejects_each_mismatch(scratch,monkeypatch,fault,part):
    out,records,root,old,_=comparison_fixture(scratch,monkeypatch)
    assert c.compare(root,records,out)['failed']==0
    if fault in ('label','void','readout'):
        path=root/'stage2/cells.json';doc=c.read_json(path)
        row=next(r for r in doc if r['task']=='ret') if fault=='readout' else doc[0]
        row[{'label':'label','void':'void','readout':'comparator_readout'}[fault]]={
            'label':'BELOW','void':['failed gate'],'readout':'center+k0'}[fault]
    elif fault=='pick':
        path=records/'stage1_om_picks.json';doc=c.read_json(path);doc['tpips/mmart']['pick']='none+k1'
    elif fault=='table':
        path=root/'stage1/tables/tpips__mmart.json';doc=c.read_json(path);doc['table']['none+k0']['map']['emotion']+=.01
    elif fault in ('range','per_seed'):
        path=records/'stage1_om_ranges.json';doc=c.read_json(path);row=doc['artelingo/qwen/emotion/ami/20']
        if fault=='range':row['r']+=.01
        else:row['per_seed'][0]+=.01
    elif fault in ('status','r17_status'):
        path=records/'status.json' if fault=='status' else old
        path.write_text(path.read_text()+'\n')
    else:
        path=next((root/'parts/turtle').glob('*/*/*/*.json'));doc=c.read_json(path);doc['input_sha256']='0'*64
    if fault not in ('status','r17_status'):put(path,doc)
    report=c.compare(root,records,out)
    assert report['failed']>0
    assert any(not item['pass'] and item['name'].startswith(part) for item in report['items'])
    if fault=='turtle':assert report['parts']['turtle_inputs']['failed']==1


def test_compare_null_effective_range_passes(scratch,monkeypatch):
    out,records,root,_,_=comparison_fixture(scratch,monkeypatch,missing_range=True)
    row=next(r for r in c.read_json(out/'c13_cells.json')['cells'] if r['id']=='SC1/mmart/qwen/emotion/ami/9/tpips')
    assert row['r']==.1 and row['r_used'] is None and row['label']=='inconclusive'
    assert c.compare(root,records,out)['failed']==0


def test_compare_low_draws_cannot_claim_full(scratch,monkeypatch):
    out,records,root,_,_=comparison_fixture(scratch,monkeypatch,low_draw=True)
    report=c.compare(root,records,out)
    assert report['failed']==0 and report['scope']=='subset'


@pytest.mark.parametrize('count',[411,413])
def test_compare_full_requires_412_turtle_sidecars(scratch,monkeypatch,count):
    out,records,root,_,_=comparison_fixture(scratch,monkeypatch,sidecars=count)
    report=c.compare(root,records,out)
    assert report['scope']=='full' and report['failed']==1
    assert next(x for x in report['items'] if x['name']=='turtle_inputs/sidecar_count')['pass'] is False


def check_range_wiring(module,scratch,monkeypatch):
    """Execute range loading with deliberately different retrieval/clustering ranges."""
    with monkeypatch.context() as patch:
        patch.setattr(module,'R17',scratch/'r17')
        put(module.R17/'records/stage1_ranges.json',{})
        patch.setattr(module,'keys',lambda d,s:['a','b','c','d'])
        patch.setattr(module,'labels',lambda d,s:{a:np.array([0,1,0,1]) for a in module.ASPECTS})
        patch.setattr(module.ns,'recorded_ranges',lambda d,e,task='ret':{a:(.13 if task=='ret' else .27) for a in module.ASPECTS})
        patch.setattr(module.ns,'arm_metrics',lambda p,y:np.zeros(4))
        work=module.Work(scratch)
        patch.setattr(work,'dependency',lambda name:{})
        patch.setattr(work,'vector',lambda *args:np.ones((4,3),dtype=np.float32))
        patch.setattr(work,'array',lambda *args:np.zeros((3,4),dtype=np.int32))
        result=work.ranges()
        assert result['artelingo/qwen/emotion/map']['r']==.13
        assert result['mmart/clip_l14/genre/map']['r']==.13
        assert result['mmart/siglip2/style/ami/20']['r']==.27
        assert result['artelingo/clip_l14/emotion/ami/20']['r']==.27


def check_unit_encoder_wiring(module,scratch,monkeypatch):
    """Picks must consume Qwen's TPIPS units and CLIP's CLAY units."""
    with monkeypatch.context() as patch:
        units={'qwen':dict(zip(module.ASPECTS,(.1,.2,.3))),
               'clip_l14':dict(zip(module.ASPECTS,(.4,.5,.6)))}
        patch.setattr(module,'keys',lambda d,s:['a','b','c','d'])
        patch.setattr(module,'labels',lambda d,s:{a:np.array([0,1,0,1]) for a in module.ASPECTS})
        patch.setattr(module.ns,'recorded_ranges',lambda d,e:units[e])
        patch.setattr(module.rt,'crossfit_all',lambda xs,lab,ros,workers:{r:{a:.2 for a in module.ASPECTS} for r in ros})
        work=module.Work(scratch)
        patch.setattr(work,'dependency',lambda name:{'sets':{'A':{'ranks':{a:49 for a in module.ASPECTS}}}})
        patch.setattr(work,'vector',lambda *args:np.ones((4,3),dtype=np.float32))
        patch.setattr(work,'clay_vector',lambda *args:np.ones((4,3),dtype=np.float32))
        patch.setattr(work,'array',lambda name,inputs,fn:fn())
        for d in module.SETS:
            assert work.pick_table(d,'tpips')['unit']==units['qwen']
            assert work.pick_table(d,'clay','A')['unit']==units['clip_l14']


def check_textset_wiring(module,scratch,monkeypatch):
    """A .02 gain picks A for retrieval and B for clustering, through to cells."""
    with monkeypatch.context() as patch:
        d='mmart';work=module.Work(scratch,clu=40,ret=100,workers=1)
        deps={f'clay:{d}:val':{'sets':{ts:{'ranks':{a:49 for a in module.ASPECTS}} for ts in ('A','B')}},
              f'pick:clay:{d}':{'text_set':'A','pick':'none+k0'},f'textset:clu:{d}':{'set':'B'},
              'counts':{},'ranges':{module.range_key(row):{'r':.1} for row in module.cell_specs()}}
        deps[f'parts:cstar:{d}:qwen']={'arrays':{f'{a}/{k}':np.zeros((5,3,4),dtype=np.int32)
            for a in module.ASPECTS for k in {module.K[d][a],20}}}
        patch.setattr(module,'keys',lambda d,s:['a','b','c','d'])
        patch.setattr(module,'labels',lambda d,s:{a:np.array([0,1,0,1]) for a in module.ASPECTS})
        patch.setattr(work,'dependency',lambda name:deps[name])
        patch.setattr(work,'pick_table',lambda d,arm,ts:dict(pick='none+k0',unit_score=.1 if ts=='A' else .12))
        patch.setattr(work,'clay_vector',lambda d,split,ts,a:np.full((4,3),.1 if ts=='A' else .12,dtype=np.float32))
        patch.setattr(work,'array',lambda name,inputs,fn:fn())
        patch.setattr(work,'stored',lambda name,inputs,fn:(None,fn()))
        patch.setattr(work,'artifact',lambda entry:entry)
        patch.setattr(module.ns,'partitions',lambda x,k,d:np.full((3,len(x)),2 if x[0,0]<.11 else 7,dtype=np.int32))
        patch.setattr(module.ns,'arm_metrics',lambda parts,y:np.array([.1 if parts[0,0]==2 else .12,0.,0.,0.]))
        assert work.pick('clay',d)['text_set']=='A'
        assert work.textset_clu(d)['set']=='B'
        stats=dict(cand=.4,comp=.3,d=.1,lo=.01,hi=.2,n_draws=100)
        def ap(dataset,encoder,arm,a,y):
            if arm=='clay':
                assert encoder=='clip_l14'
                assert np.allclose(work.vector(dataset,module.HELD[dataset],encoder,arm,a),.1)
            return np.full((5,4) if arm=='cstar' else (4,),.4 if arm=='cstar' else .3,dtype=np.float64),'none+k0'
        patch.setattr(work,'ap',ap)
        patch.setattr(module.ns,'paired_retrieval',lambda *args,**kwargs:stats)
        assert len(work.cells('SC4',d,'qwen'))==3
        deps[f'parts:clay:{d}']=work.partitions_section('clay',d)
        def joint(candidate,comparator,y,count,workers):
            assert np.all(comparator==7), 'clustering cells must consume text set B'
            return {m:{**stats,'n_draws':count} for m in module.METRICS}
        patch.setattr(module.ns,'joint_clustering',joint)
        assert len(work.cells('SC3',d,'qwen'))==14


@pytest.mark.parametrize('check',[check_range_wiring,check_unit_encoder_wiring,check_textset_wiring],ids=['range_routes','unit_encoder','textset_routes'])
def test_behavioral_wiring(scratch,monkeypatch,check):
    check(c,scratch,monkeypatch)


MUTATIONS=[
    ('range_ret_clu',"'ret' if m=='map' else 'clu'","'clu' if m=='map' else 'ret'",check_range_wiring),
    ('unit_encoder',"encoder='qwen' if arm=='tpips' else 'clip_l14'","encoder='clip_l14' if arm=='tpips' else 'qwen'",check_unit_encoder_wiring),
    ('pick_margin',"ts=choose_textset(scores,'ret')","ts=choose_textset(scores,'clu')",check_textset_wiring),
    ('clustering_margin',"return dict(set=choose_textset(scores,'clu'),scores=scores,per_aspect=per_aspect)",
        "return dict(set=choose_textset(scores,'ret'),scores=scores,per_aspect=per_aspect)",check_textset_wiring),
    ('retrieval_textset',"ts=self.dependency(f'pick:clay:{d}')['text_set']",
        "ts=self.dependency(f'textset:clu:{d}')['set']",check_textset_wiring),
    ('clustering_textset',"ts=self.dependency(f'textset:clu:{d}')['set'] if arm=='clay' else None",
        "ts=self.dependency(f'pick:clay:{d}')['text_set'] if arm=='clay' else None",check_textset_wiring),
]


@pytest.mark.parametrize('name,before,after,check',MUTATIONS,ids=[row[0] for row in MUTATIONS])
def test_wiring_catches_temporary_mutation(scratch,monkeypatch,name,before,after,check):
    source=Path(c.__file__).read_text()
    assert source.count(before)==1
    check(c,scratch/'baseline',monkeypatch)
    path=scratch/'mutant.py'
    mutated=source.replace(before,after,1)
    compile(mutated,str(path),'exec')
    path.write_text(mutated)
    spec=importlib.util.spec_from_file_location('c13_mutant_'+scratch.name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    with pytest.raises(AssertionError):check(module,scratch/'mutated',monkeypatch)
    put(scratch/'mutation_result.json',{'mutation':name,'caught':True,'check':check.__name__})
    assert Path(c.__file__).read_text()==source
