"""Quick independent-rule checks. All scratch files stay under c13_smoke.

Run with -p no:cacheprovider and --basetemp under c13_smoke to keep pytest's
own artifacts within the same boundary. CPU only. The real runner is loaded
only in a separate checker-contract subprocess, never by the C13 import.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pp_c13 as c


@pytest.fixture
def scratch():
    p=c.SMOKE_OUT/'unit'/uuid.uuid4().hex
    p.mkdir(parents=True)
    return p


def all_cells():
    result=[]
    for sc,e,a,t,d in sorted(c.expected_ids()):
        result.append(dict(sc=sc,encoder=e,aspect=a,task=t,data=d,candidate='candidate',comparator='bar',lo=.01,r=.1 if sc in ('SC6','SC10','SC14') else None))
    return result


def all_cost():
    return [dict(sc='SC15',encoder=e,cstar_ms_max=1.,teacher_ms_min=2.) for e in c.ENCODERS]


def test_set_rule_strict_cut_unknown_tiers_md5_and_hash_order():
    keys=c.hashed_order(['a','b','c','d','e','f','g'])
    rows=pd.DataFrame(dict(image_file=keys,author=['NEW, Ada']*2+['UNKNOWN MASTER','VAN GOGH, Vincent','GO?H, Other','NEW, Ada','NEW, Bea'],type=['genre']*7,max_cos=[.94,.90,.1,.1,.1,.95,.949999]))
    result=c.semart_rule(rows,{'vincent-van-gogh'},lambda k:'duplicate' if k in keys[:2] else k)
    assert result['gate']['paintings']==3
    assert result['n_set']==2 and result['keys']==[keys[0],keys[6]]
    assert result['md5_groups']==[dict(md5='duplicate',keep=keys[0],dropped=[keys[1]])]
    assert result['keys_sha256']['set']==hashlib.sha256('\n'.join([keys[0],keys[6]]).encode()).hexdigest()
    assert result['close_before_md5']==2 and result['close']==2


@pytest.mark.parametrize('author,tier',[('RAFFAELLO Sanzio','exact'),('GOGH, Vincent','given_surname'),('GOGH, Other','surname_only'),('NEW, Ada','none'),('GO?H, Vincent','given_surname')])
def test_name_rule(author,tier):
    assert c.artist_tier(author,{'vincent-van-gogh','raphael'})==tier


def test_saved_tier_mismatch_refused():
    t=pd.DataFrame(dict(image_file=['a'],author=['NEW, Ada'],type=['genre'],max_cos=[.1],artist_tier=['exact']))
    with pytest.raises(AssertionError,match='tiers'):c.semart_rule(t,set(),lambda k:k)


def test_twice_normalisation_preserves_float32_rounding():
    rng=np.random.default_rng(2);raw=rng.normal(size=(32,31)).astype(np.float16)
    once=raw.astype(np.float32);once/=np.maximum(np.linalg.norm(once,axis=1,keepdims=True),1e-12)
    twice=once/np.linalg.norm(once,axis=1,keepdims=True)
    assert not np.array_equal(once,twice)
    assert np.array_equal(c.store_retrieval(raw),twice)
    assert c.store_retrieval(raw).dtype==np.float32


@pytest.mark.parametrize('lo,r,want',[(1e-12,None,'WIN'),(0.,None,'not WIN'),(0.,.1,'LEVEL'),(-.099,.1,'LEVEL'),(-.1,.1,'BELOW'),(-.101,.1,'BELOW'),(0.,0.,'BELOW')])
def test_outcome_boundaries(lo,r,want):
    assert c.cell_outcome(lo,r)==want


@pytest.mark.parametrize('sc,d,want',[('SC1','art','fails'),('SC2','art','fails'),('SC3','art','qualified'),('SC4','aff','fails'),('SC5','aff','fails'),('SC7','mm','fails'),('SC9','mm','qualified'),('SC11','sem','fails'),('SC13','sem','qualified')])
def test_verdict_conjunctions(sc,d,want):
    rows=all_cells();result=c.decisions(rows,all_cost(),c.good_status())
    assert all(v=='holds' for sets in result['verdicts'].values() for v in sets.values())
    next(x for x in rows if x['sc']==sc and x['encoder']=='clip_b32')['lo']=0.
    assert c.decisions(rows,all_cost(),c.good_status())['verdicts']['clip_b32'][d]==want


def test_cost_strict_inequality_and_missing_cells():
    cost=all_cost();cost[-1]['cstar_ms_max']=2.
    assert c.decisions(all_cells(),cost,c.good_status())['verdicts'][c.ENCODERS[-1]]['cost']=='fails'
    with pytest.raises(AssertionError,match='missing live'):c.decisions(all_cells()[1:],cost,c.good_status())
    c.decisions(all_cells()[1:],cost,c.good_status(),partial=True)


@pytest.mark.parametrize('d', ['art','aff','mm','sem','cost'])
def test_failed_cstar_voids_all_dependent_paths(d):
    s=c.good_status();s['runs']['clip_b32'].update(cstar='failed',photo='failed')
    assert c.voids(s,'clip_b32',d)
    assert not c.voids(s,'clip_b16',d)


def test_matched_void_preserves_teacher_and_photos():
    s=c.good_status();s['gates']['v']['clip_b32']='failed'
    for d,sc in [('art','SC3'),('mm','SC9'),('sem','SC13')]:
        assert c.voids(s,'clip_b32',d) and c.voids(s,'clip_b32',d,sc,'clu')
        assert not c.voids(s,'clip_b32',d,'SC6','clu')
    assert not c.voids(s,'clip_b32','aff')


def test_vii_voids_retrieval_preserves_ami_claims_and_old_semart():
    s=c.good_status();s['gates']['vii']['artelingo']='failed'
    assert c.voids(s,'clip_b32','sem')
    assert c.voids(s,'clip_b32','art','SC6','ret')
    assert not c.voids(s,'clip_b32','art','SC6','clu')
    assert not c.voids(s,'qwen','sem')
    assert c.voids(s,'qwen','sem','SC14','ret')
    assert not c.voids(s,'qwen','sem','SC14','clu')


def test_store_and_parity_gate_scope():
    s=c.good_status();s['gates']['i_art_parity']['siglip1']='failed'
    assert c.voids(s,'siglip1','sem') and not c.voids(s,'clip_b32','sem')
    s=c.good_status();s['gates']['i_semart_parity']['qwen']='failed'
    assert all(c.voids(s,e,'sem') for e in c.ENCODERS)
    assert not c.voids(s,'clip_b32','art')
    s=c.good_status();s['gates']['i_semart_genre_word']='failed'
    assert not c.voids(s,'qwen','sem') and c.voids(s,'qwen','sem','SC14','ret')


def test_inconclusive_never_qualified():
    s=c.good_status();s['gates']['iv']='failed'
    result=c.decisions([],all_cost(),s)
    assert all(v=='inconclusive' for sets in result['verdicts'].values() for d,v in sets.items() if d!='cost')


@pytest.mark.parametrize('e',c.ENCODERS)
def test_d17_claim_licensing(e):
    assert c.teacher_claim(e,'sem','ret','WIN')==('better than' if e in c.ENCODERS[:4] else 'level with or better than')
    assert c.teacher_claim(e,'sem','clu','WIN')=='level with or better than'
    assert c.teacher_claim(e,'art','ret','WIN')=='level with or better than'
    assert c.teacher_claim(e,'sem','ret','LEVEL')=='level with or better than'
    assert c.teacher_claim(e,'sem','ret','BELOW') is None
    assert c.teacher_claim(e,'sem','ret','WIN',True)=='inconclusive'


def test_status_exact_schema():
    for mutation in ('missing','extra','pending'):
        s=c.good_status()
        if mutation=='missing':s['gates'].pop('iv')
        elif mutation=='extra':s['gates']['unknown']='ok'
        else:s['gates']['iv']='pending'
        with pytest.raises(AssertionError):c.validate_status(s)


def toy_probe(root):
    batches=root/'probe/batches';batches.mkdir(parents=True)
    passes=dict(meta=dict(n_images=2),passes={},repeats_discarded=[dict(pass_='qwen',file='discarded/bad.json')])
    passes['repeats_discarded'][0]['pass']=passes['repeats_discarded'][0].pop('pass_')
    discarded=root/'probe/discarded';discarded.mkdir()
    (discarded/'bad.json').write_text('{"pass":"qwen","attempt":0,"discarded":true}')
    names=list(c.ENCODERS)+['heads_'+e for e in c.ENCODERS]+['qwen_'+c.TEACHER[a] for a in c.ASPECTS]+['teacher']
    for name in names:
        heads=name.startswith('heads_');teacher=name.startswith('qwen_')
        repeats={} if name=='teacher' else {str(r):dict(file=f'{name}__r{r}.json') for r in range(3)}
        entry=dict(encoder=name,kind='heads' if heads else 'model',repeats=repeats)
        if not heads:entry['e2e']=dict(file=name+'__e2e.json')
        passes['passes'][name]=entry
        refs={**repeats,**({'e2e':entry['e2e']} if 'e2e' in entry else {})}
        for rep,ref in refs.items():
            # c* max is 9 ms; teacher min is 6 ms. Means suggest the wrong answer.
            ms=(1 if heads else ([1,2,8][int(rep)] if rep!='e2e' else 3))
            if teacher:ms=([2,4,10][int(rep)] if rep!='e2e' else 4)
            if name=='teacher':ms=12
            raw=dict(pass_=name,repeat=None if rep=='e2e' else int(rep),n_images=2,batch_n=[1,1],batch_seconds=[ms/1000,ms/1000],batch_size=32,warmup_batches=3)
            raw['pass']=raw.pop('pass_');(batches/ref['file']).write_text(json.dumps(raw))
    (root/'probe/passes.json').write_text(json.dumps(passes))
    return passes


def test_cost_raw_repeat_arithmetic(scratch):
    toy_probe(scratch);result=c.raw_cost(scratch,prep_meta={});cell=result['cells'][0]
    assert cell['cstar_ms']==[2.,3.,9.]
    assert cell['teacher_ms']==[6.,12.,30.]
    assert cell['cstar_ms_max']==9 and cell['teacher_ms_min']==6
    assert cell['cstar_ms_mean']<cell['teacher_ms_mean']
    assert cell['cstar_ms_e2e']==4 and cell['teacher_ms_e2e']==12
    assert cell['repeats_discarded']==1
    assert result['gate']['qwen_default_e2e_ms']==3
    assert result['table']['whole_set_16384']['qwen']['cstar_s']==pytest.approx(np.mean([2,3,9])*16.384)


def test_cost_refuses_discarded_timed_file(scratch):
    toy_probe(scratch)
    # The accepted name is different: detect a copied discarded payload.
    (scratch/'probe/discarded/other_name.json').write_bytes((scratch/'probe/batches/qwen__r0.json').read_bytes())
    with pytest.raises(AssertionError,match='discarded'):c.raw_cost(scratch,prep_meta={})


def test_tolerance_and_exact_categories():
    cmp=c.Comparison()
    for args in [('boundary',0.,1e-6),('beyond',0.,1.00001e-6),('nan',.1,float('nan')),('missing',.1,None),('bool',True,1),('pick','none+k0','center+k0')]:cmp.add(*args)
    assert [x['pass'] for x in cmp.items]==[True,False,False,False,False,False]


def test_merge_missing_and_differing_draw_counts(scratch,monkeypatch):
    with pytest.raises(ValueError,match='missing sections'):c.merge(scratch)
    p=scratch/'sections';p.mkdir()
    base=dict(complete=True,root=str(c.DEFAULT_ROOT),draws_clu=2000,draws_ret=10000,kmeans_threads=8,source_sha256='x',inputs={},result={})
    for name,draws in [('set',2000),('cost',40)]:
        (p/f'{name}.json').write_text(json.dumps(dict(base,name=name,draws_clu=draws)))
    with pytest.raises(ValueError,match='draw counts'):c.assemble(scratch,False)


@pytest.mark.parametrize('change,reason',[({'scope':'subset:set'},'subset'),({'kmeans_threads':1},'threads'),({'draws_clu':40},'draws'),({},None)])
def test_certificate_refusals(change,reason):
    doc=dict(scope='full',kmeans_threads=8,draws_clu=2000,draws_ret=10000,
             results={'set':dict(n_set=11159,gate={'pass':True}),'cost':{'gate':{'pass':True}}},
             status=c.good_status(),cost=[dict(encoder=e,n_images=2000) for e in c.ENCODERS])
    doc.update(change)
    rows=[dict(x,n_draws=2000 if x['task']=='clu' else 10000,n_rows=dict(artelingo_held=12281,affection_test=8501,mmart_test=16384,semart_new=11159)[x['data']]) for x in all_cells()]
    refusal=c.record_refusal(doc,{'pass':True},rows)
    assert (reason in refusal) if reason else refusal is None
    assert c.record_refusal(dict(doc,scope='full'),{'pass':False},rows)=='comparison failed'


def test_record_keys_equal_c13_check_contract(scratch):
    root=scratch/'root';records=scratch/'records'
    files=['stage2/cells_artelingo_clu.json','stage2/cells_semart_ret.json','stage2/cells_x.pending.json','stage3/cells_cost.json','stage3/cost_table.json','probe/passes.json','probe/batches/x.json','probe/discarded/x.json']
    for name in files:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    records.mkdir()
    for name in ('status.json','semart_set.json','stage1_clip_b32.json','stage1_semart.json','c13.json'):(records/name).write_text('{}')
    manifest={name:c.file_sha(p) for name,p in c.result_paths(root,records).items()}
    (records/'c13.json').write_text(json.dumps({'pass':True,'sha256':manifest}))
    result=run_python('''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import pp_run
root,records=map(Path,sys.argv[2:4])
pp_run.OUT=root
pp_run.PP.SMOKE=False
pp_run.PP.record_path=lambda name:records/name
pp_run.PP.read_committed_record=lambda name:json.loads((records/name).read_text())
saved=json.loads((records/'c13.json').read_text())
pp_run.c13_check()
for key in saved['sha256']:
    reduced={'pass':True,'sha256':{k:v for k,v in saved['sha256'].items() if k!=key}}
    (records/'c13.json').write_text(json.dumps(reduced))
    try:pp_run.c13_check()
    except AssertionError:pass
    else:raise AssertionError('Extraneous manifest key: '+key)
(records/'c13.json').write_text(json.dumps(saved))
pp_run.c13_check()
''',c.ROUND,root,records,cwd=scratch)
    assert result.returncode==0,result.stderr


def test_dependencies_refuse_absence(scratch):
    with pytest.raises(ValueError,match='missing dependency'):c.Work(out=scratch).dep('picks:clip_b32:art')


def test_output_boundary():
    with pytest.raises(ValueError):c.check_output(c.R14/'c13')


@pytest.mark.parametrize('scope',['subset:set','full'])
def test_compare_refuses_certificate_and_invalidates_old_pass(scope,scratch,monkeypatch):
    doc,root,records,out=full_compare_fixture(scratch,monkeypatch)
    (records/'c13.json').write_text('{"pass":true}')
    doc['scope']=scope
    if scope=='full':
        p=root/'stage2/cells_artelingo_clu.json';rows=c.read_json(p)
        rows[0]['candidate_value']+=.001;p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='subset|failed'):
        c.compare(root,records,out,write_record=True)
    assert c.read_json(records/'c13.json')['pass'] is False
    report=c.read_json(out/'compare.json')
    assert report['record'].startswith('refused:')
    failures=[x for x in report['items'] if not x['pass']]
    if scope=='full':
        assert len(failures)==1 and failures[0]['name'].endswith('/candidate_value')
    else:assert not failures


def test_unrounded_prep_metadata_and_network_parameters(scratch):
    passes=toy_probe(scratch)
    passes['passes']['heads_qwen']['params_network']=[4197376]*3
    (scratch/'probe/passes.json').write_text(json.dumps(passes))
    result=c.raw_cost(scratch,prep_meta={'qwen_default':dict(seconds=1448.5740305329673,n=16782)})
    row=result['table']['prep_log_qwen']['qwen_default']
    assert row['seconds']==1448.5740305329673
    assert row['ms_per_image']==1448.5740305329673/16782*1000
    assert result['table']['passes']['heads_qwen']['params_network']==[4197376]*3


def test_check_outcomes_detects_missing_verdict_and_numeric_change(scratch,monkeypatch):
    status=c.good_status();rows=[]
    for row in all_cells():
        row.update(candidate_value=.2,comparator_value=.1,d=.1,hi=.02,n_rows=11159,n_draws=2000 if row['task']=='clu' else 10000,seeds=list(c.SEEDS))
        rows.append(row)
    cost=all_cost();counts=dict(mmart_n=16384,semart_n=11159,semart_artists=2717)
    expected=c.decisions(rows,cost,status,counts=counts)
    doc=dict(scope='full',decisions=expected,status=status,cost=cost,draws_clu=2000,draws_ret=10000,kmeans_threads=8)
    c.atomic_json(scratch/'derived.json',doc)
    root=scratch/'round';(root/'stage4').mkdir(parents=True)
    actual=dict(status=status,cells=expected['cells'],cost=cost,claims=expected['claims'],verdicts=expected['verdicts'],counts=counts)
    # Use the round's documented set labels in its verdict tree.
    aliases=dict(art='artelingo',aff='photos',mm='mmart',sem='semart',cost='cost')
    actual['verdicts']={e:{aliases[d]:v for d,v in sets.items()} for e,sets in expected['verdicts'].items()}
    p=root/'stage4/outcomes.json';p.write_text(json.dumps(actual))
    assert c.check_outcomes(root,scratch)['pass']
    actual['cells'][0]['d']=.123
    p.write_text(json.dumps(actual))
    assert not c.check_outcomes(root,scratch)['pass']
    actual['cells'][0]['d']=.1
    actual['verdicts']['clip_b32'].pop('semart')
    p.write_text(json.dumps(actual))
    assert not c.check_outcomes(root,scratch)['pass']


def test_export_sidecars_and_key_join(scratch):
    p=scratch/'export.npz';x=np.array([[1,2],[3,4]],np.float16)
    np.savez_compressed(p,keys=np.array(['b','a']),emb=x)
    side=dict(sha256=c.file_sha(p),keys_sha256=c.keys_sha(['b','a']))
    p.with_suffix('.json').write_text(json.dumps(side))
    work=c.Work(out=scratch)
    assert np.array_equal(work.export(p,['a','b']),x[[1,0]].astype(np.float32))
    p.with_suffix('.pp.json').write_text(json.dumps({'sha256':'wrong'}))
    with pytest.raises(AssertionError,match='checksum'):work.export(p,['a','b'])


def test_cache_binds_array_bytes_and_thread_count(scratch,monkeypatch):
    work=c.Work(out=scratch);calls=[]
    monkeypatch.setattr(c.mm,'kmeans_partitions',lambda x,n:calls.append(n) or np.full((3,len(x)),n,np.int32))
    x=np.eye(3,dtype=np.float32);keys=['a','b','c']
    assert (work.parts(x,keys)==8).all()
    assert (work.parts(x,keys)==8).all() and calls==[8]
    work.threads=1
    assert (work.parts(x,keys)==1).all() and calls==[8,1]
    work.parts(x[:,::-1].copy(),keys)
    assert calls==[8,1,1]


def test_no_forbidden_direct_imports():
    result=run_python('''
import sys
sys.path.insert(0,sys.argv[1])
import pp_c13
forbidden={'bs_artclu','bs_artret','bs_emoset_score','bs_outcome','bs_cstar','bs_affection','bs_emoset','bs_photo',
           'mt_clu','mt_ret','mt_outcome','mt_export','mt_matched','mt_data','mt_run'}
assert not set(sys.modules)&forbidden,sorted(set(sys.modules)&forbidden)
assert not [n for n in sys.modules if n.startswith('pp_') and n!='pp_c13']
''',c.ROUND)
    assert result.returncode==0,result.stderr


def test_affection_cell_layout_omits_pick_annotations(scratch,monkeypatch):
    root=scratch/'root';records=root/'records';out=scratch/'cmp'
    records.mkdir(parents=True);(root/'stage2').mkdir()
    status=c.good_status();(records/'status.json').write_text(json.dumps(status))
    row=dict(sc='SC5',encoder='clip_b32',aspect='emotion',task='ret',data='affection_test',candidate='photo_clip_b32',comparator='frozen_clip_b32',
             candidate_value=.2,comparator_value=.1,d=.1,lo=.08,hi=.12,r=None,outcome='WIN',n_draws=10000,seeds=list(c.SEEDS),n_rows=8501,
             candidate_readout='center+k10',comparator_readout='center+k10',n_queries=8501)
    saved={k:v for k,v in row.items() if k not in ('outcome','candidate_readout','comparator_readout','n_queries')}
    path=root/'stage2/cells_affection.json';path.write_text(json.dumps([saved]))
    doc=dict(root=str(root),scope='subset:smoke:clip_b32:aff',draws_clu=2000,draws_ret=10000,kmeans_threads=8,source_sha256='x',inputs={},
             results={'smoke:clip_b32:aff':[row]},cells=[row],cost=[],status=status)
    monkeypatch.setattr(c,'source_identity',lambda:'x');monkeypatch.setattr(c,'assemble',lambda *args:doc)
    monkeypatch.setattr(c,'result_paths',lambda *args:{})
    assert c.compare(root,records,out)['pass']
    saved['candidate_readout']='none+k0';path.write_text(json.dumps([saved]))
    assert not c.compare(root,records,out)['pass']


def test_discrete_seed_and_population_fields_are_exact():
    cmp=c.Comparison();cmp.add('seed',0,1e-8);cmp.add('rows',8501,8501.0);cmp.add('rows/ok',8501,8501)
    assert [x['pass'] for x in cmp.items]==[False,False,True]


def test_legacy_array_digest_and_npz_digest_are_both_checked(scratch):
    p=scratch/'held.npz';x=np.array([[1,2],[3,4]],np.float16);np.savez_compressed(p,img=x)
    sha=c.frozen.rs_vh.array_sha(x)
    side=dict(sha256=sha,list_sha256=c.frozen.rs_vh.names_sha(['b','a']),arrays={'img':dict(shape=[2,2],dtype='float16',sha256=sha)})
    p.with_suffix('.json').write_text(json.dumps(side))
    p.with_suffix('.pp.json').write_text(json.dumps({'sha256':c.file_sha(p)}))
    work=c.Work(out=scratch)
    assert np.array_equal(work.export(p,['a','b'],'img',['b','a']),x[[1,0]].astype(np.float32))
    side['arrays']['img']['sha256']='wrong';p.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError,match='array checksum'):work.export(p,['a','b'],'img',['b','a'])


def test_positional_export_requires_key_list_hash(scratch):
    p=scratch/'unbound.npz';np.savez_compressed(p,img=np.eye(2,dtype=np.float16))
    p.with_suffix('.json').write_text(json.dumps({'sha256':c.file_sha(p)}))
    with pytest.raises(AssertionError,match='key-list identity'):c.Work(out=scratch).export(p,['a','b'],'img',['a','b'])


def test_legacy_matched_probability_array_digest(scratch):
    p=scratch/'matched.npz';prob=np.array([[.1,.9],[.7,.3]],np.float32);np.savez_compressed(p,prob=prob)
    p.with_suffix('.json').write_text(json.dumps(dict(sha256=c.frozen.rs_vh.array_sha(prob),list_sha256=c.frozen.rs_vh.names_sha(['a','b']),rows=2,width=2,dtype='float32')))
    assert np.array_equal(c.Work(out=scratch).export(p,['b','a'],'prob',['a','b']),prob[[1,0]])


def run_python(source,*args,cwd=None):
    env=os.environ.copy()
    for key in ('PP_SMOKE','PP_OUTPUTS'):env.pop(key,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
    return subprocess.run([sys.executable,'-c',source,*map(str,args)],cwd=cwd or c.SMOKE_OUT,
                          env=env,capture_output=True,text=True,timeout=60)


def full_compare_fixture(scratch,monkeypatch):
    root=scratch/'round';records=root/'records';out=scratch/'comparison'
    records.mkdir(parents=True);status=c.good_status()
    (records/'status.json').write_text(json.dumps(status))
    rows=[]
    populations=dict(artelingo_held=12281,affection_test=8501,mmart_test=16384,semart_new=11159)
    for row in all_cells():
        row.update(candidate_value=.2,comparator_value=.1,d=.1,hi=.12,n_rows=populations[row['data']],
                   n_draws=2000 if row['task']=='clu' else 10000,seeds=list(c.SEEDS),outcome='WIN')
        rows.append(row)
    for part,data,task in [('artelingo_clu','artelingo_held','clu'),('artelingo_ret','artelingo_held','ret'),
                           ('affection','affection_test',None),('mmart_clu','mmart_test','clu'),('mmart_ret','mmart_test','ret'),
                           ('semart_clu','semart_new','clu'),('semart_ret','semart_new','ret')]:
        p=root/f'stage2/cells_{part}.json';p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps([r for r in rows if r['data']==data and (task is None or r['task']==task)]))
    costs=[dict(row,n_images=2000) for row in all_cost()]
    results={'set':dict(n_set=11159,gate={'pass':True}),
             'cost':dict(cells=costs,table={},gate={'pass':True})}
    for p,value in [(records/'semart_set.json',results['set']), (records/'stage1_fixture.json',{}),
                    (root/'stage3/cells_cost.json',costs),(root/'stage3/cost_table.json',{}),
                    (root/'stage0/gate_viii.json',{'pass':True}),(root/'probe/passes.json',{}),
                    (root/'probe/batches/fixture.json',{}),(root/'probe/discarded/fixture.json',{})]:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    doc=dict(root=str(root),scope='full',sections=list(c.FULL_SECTIONS),draws_clu=2000,draws_ret=10000,kmeans_threads=8,
             source_sha256='fixture',inputs={},results=results,cells=rows,cost=costs,status=status,
             derived_gates=dict(vi=True,viii=True))
    monkeypatch.setattr(c,'source_identity',lambda:'fixture')
    monkeypatch.setattr(c,'assemble',lambda *args,**kwargs:doc)
    return doc,root,records,out


def test_write_record_happy_path(scratch,monkeypatch):
    _,root,records,out=full_compare_fixture(scratch,monkeypatch)
    expected={name:c.file_sha(p) for name,p in c.result_paths(root,records).items()}
    assert c.main(['compare','--root',str(root),'--records',str(records),'--out',str(out),'--write-record'])==0
    record=c.read_json(records/'c13.json')
    assert record['pass'] is True and record['scope']=='full'
    assert record['sha256']==expected
    assert (out/'derived.json').exists() and not (out/'derived_subset.json').exists()


@pytest.mark.parametrize('d',['art','aff','mm','sem'])
@pytest.mark.parametrize('split',['val','test'])
def test_frozen_normalisation_per_set(d,split,scratch,monkeypatch):
    rng=np.random.default_rng(3);raw=rng.normal(size=(12,31)).astype(np.float16)
    keys=[f'key{i}' for i in range(len(raw))];work=c.Work(out=scratch)
    monkeypatch.setattr(work,'population',lambda *args:(keys,{}))
    monkeypatch.setattr(work,'store',lambda *args:raw.astype(np.float32))
    monkeypatch.setattr(work,'track',lambda p:p)
    if d=='mm':
        table=pd.DataFrame(dict(key=keys,artelingo=['none']*len(keys),painting=keys))
        monkeypatch.setattr(c.pd,'read_parquet',lambda p:table)
    actual=work.vector('clip_b32',d,'frozen',split,'emotion',retrieval=True)
    expected=c.ac_data._unit(raw) if d in ('art','aff') else c.store_retrieval(raw)
    assert np.array_equal(actual,expected)
    assert not np.array_equal(c.ac_data._unit(raw),c.store_retrieval(raw))


@pytest.mark.parametrize('d',['art','aff'])
def test_teacher_uses_saved_per_query_ap(d,scratch,monkeypatch):
    work=c.Work(out=scratch);y=np.array([0,1,0],np.int64)
    monkeypatch.setattr(work,'population',lambda *args:(['a','b','c'],{'emotion':y}))
    monkeypatch.setattr(work,'get_pick',lambda *args:'center+k10')
    monkeypatch.setattr(work,'json',lambda p:{'affection':{'teacher_wording':{'retrieval':'artwork'}}})
    monkeypatch.setattr(work,'track',lambda p:p)
    monkeypatch.setattr(work,'vector',lambda *args,**kw:pytest.fail('teacher retrieval must not recompute vectors'))
    seen=[]
    def saved(path,labels):
        assert np.array_equal(labels,y);seen.append(path)
        return np.array([.4,.2,.8],np.float64)
    monkeypatch.setattr(c.frozen,'saved_ap',saved)
    ap,pick=work.retrieval('clip_b32',d,'teacher','emotion')
    assert pick=='center+k10' and np.array_equal(ap,[.4,.2,.8])
    assert seen==[c.R11/'stage5/perquery/instructed__emotion__hemotion__s-.npz' if d=='art' else
                  c.R12/'stage2/perquery/affection__teacher_artwork__emotion__s-.npz']


@pytest.mark.parametrize('gate,family',[('ii',None),('iii','r10'),('iii','r14')])
def test_ii_iii_void_new_cost_and_every_set(gate,family):
    status=c.good_status()
    if family:status['gates'][gate][family]='failed'
    else:status['gates'][gate]='failed'
    result=c.decisions(all_cells(),all_cost(),status)
    for e in c.NEW:
        assert all(v=='inconclusive' for v in result['verdicts'][e].values())
        assert c.voids(status,e,'cost','SC15')
        assert next(x for x in result['cost'] if x['encoder']==e)['status']=='inconclusive'
    if family:
        affected=('qwen',) if family=='r10' else c.ENCODERS[1:4]
        for e in affected:
            assert result['verdicts'][e]['sem']=='inconclusive'
            assert c.voids(status,e,'sem','SC14','clu') and c.voids(status,e,'sem','SC14','ret')
            assert result['verdicts'][e]['cost']=='holds'


def test_iv_and_viii_have_different_cost_scope():
    status=c.good_status();status['gates']['iv']='failed'
    assert not any(c.voids(status,e,'cost','SC15') for e in c.ENCODERS)
    status=c.good_status();status['gates']['viii']='failed'
    result=c.decisions(all_cells(),all_cost(),status)
    assert all(v['cost']=='inconclusive' for v in result['verdicts'].values())
    assert all(row['status']=='inconclusive' for row in result['cost'])
    assert not c.voids(status,'clip_b32','art')


def test_per_set_store_failure():
    status=c.good_status();status['gates']['i_store']['clip_b32']['affection']='failed'
    result=c.decisions(all_cells(),all_cost(),status)
    assert result['verdicts']['clip_b32']['aff']=='inconclusive'
    assert result['verdicts']['clip_b32']['art']==result['verdicts']['clip_b32']['mm']==result['verdicts']['clip_b32']['sem']=='holds'
    assert all(x['claim']=='inconclusive' for x in result['claims'] if x['encoder']=='clip_b32' and x['data']=='affection_test')


@pytest.mark.parametrize('gate,d,sc',[('vi','sem','SC14'),('viii','cost','SC15')])
@pytest.mark.parametrize('measured,recorded',[(False,'ok'),(False,'failed'),(True,'failed')])
def test_derived_gate_failure_or_disagreement_voids(gate,d,sc,measured,recorded,scratch,monkeypatch):
    status=c.good_status();status['gates'][gate]=recorded
    result=c.decisions(all_cells(),all_cost(),status,derived_gates={gate:measured})
    assert all(sets[d]=='inconclusive' for sets in result['verdicts'].values())
    assert all(c.voids(status,e,d,sc,'ret',{gate:measured}) for e in c.ENCODERS)
    doc,_,_,_=full_compare_fixture(scratch,monkeypatch)
    doc['results']['set' if gate=='vi' else 'cost']['gate']['pass']=measured
    doc['status']['gates'][gate]=recorded
    assert c.record_refusal(doc,{'pass':True},doc['cells']) is not None


def test_discard_log_filtered_by_encoder(scratch):
    passes=toy_probe(scratch)
    passes['repeats_discarded']=[{'pass':'qwen','file':'discarded/bad.json'},
                               {'pass':'heads_clip_b32','file':'discarded/heads_bad.json'},
                               {'pass':'qwen_emotion_focus','file':'discarded/teacher_bad.json'}]
    (scratch/'probe/passes.json').write_text(json.dumps(passes))
    rows={row['encoder']:row for row in c.raw_cost(scratch,prep_meta={})['cells']}
    assert rows['qwen']['repeats_discarded']==2
    assert rows['clip_b32']['repeats_discarded']==2
    assert rows['clip_b16']['repeats_discarded']==1
    assert [x['pass'] for x in rows['clip_b16']['repeats_discarded_log']]==['qwen_emotion_focus']


def test_discarded_hard_link_is_refused(scratch):
    toy_probe(scratch)
    os.link(scratch/'probe/batches/clip_b32__r0.json',scratch/'probe/discarded/unrelated_filename.json')
    with pytest.raises(AssertionError,match='discarded'):c.raw_cost(scratch,prep_meta={})


def test_export_requires_digest(scratch):
    path=scratch/'unchecked.npz';np.savez_compressed(path,keys=np.array(['a','b']),emb=np.eye(2,dtype=np.float16))
    path.with_suffix('.json').write_text(json.dumps({'keys_sha256':c.keys_sha(['a','b'])}))
    with pytest.raises(AssertionError,match='no checked digest'):c.Work(out=scratch).export(path,['a','b'])


def test_unknown_prefix_is_literal():
    table=pd.DataFrame(dict(image_file=['a','b','c'],author=['UNKNOWN MASTER','Unknown Master','unknown master'],type=['genre']*3,max_cos=[.1]*3))
    result=c.semart_rule(table,set(),lambda k:k)
    assert result['keys']==c.hashed_order(['b','c'])


@pytest.mark.parametrize('token,value,want',[('4.7',4.72,True),('4.8',4.72,False),('4.72',4.72,True),
                                            ('11,159',11159,True),('32',32,True),('4.70',4.72,False)])
def test_sentence_printed_precision(token,value,want):
    assert c.sentence_number_matches(token,value)==want


def test_subset_compare_preserves_full_derived(scratch,monkeypatch):
    doc,root,records,out=full_compare_fixture(scratch,monkeypatch);doc['scope']='subset:fixture'
    out.mkdir();(out/'derived.json').write_bytes(b'previous merged full bytes\n')
    assert c.compare(root,records,out)['pass']
    assert (out/'derived.json').read_bytes()==b'previous merged full bytes\n'
    assert c.read_json(out/'derived_subset.json')['scope']=='subset:fixture'


def golden_outcomes_fixture(scratch):
    # Independently authored outcome JSON: never copied from decisions/derived.
    actual=json.loads('''{
      "cells":[{"sc":"SC14","encoder":"qwen","aspect":"genre","task":"ret","data":"semart_new",
        "candidate":"cstar_qwen","comparator":"teacher_genre","candidate_value":0.3,"comparator_value":0.2,
        "d":0.1,"lo":0.08,"hi":0.12,"r":0.02,"outcome":"WIN","status":"decided","void":[]}],
      "claims":[{"sc":"SC14","encoder":"qwen","aspect":"genre","task":"ret","data":"semart_new",
        "outcome":"WIN","status":"decided","void":[],"claim":"better than"}],
      "cost":[{"sc":"SC15","encoder":"qwen","cstar_ms_max":4.72,"teacher_ms_min":10.06,
        "cstar_ms_mean":4.61,"teacher_ms_mean":10.14,"cstar_ms_e2e":5.24,"teacher_ms_e2e":11.76}],
      "verdicts":{"qwen":{"semart":{"verdict":"holds","void":[],
        "licensed":"On 11,159 paintings by 2,717 artists, frozen c* on Qwen3-VL-Embedding-2B clusters and retrieves by type."},
        "cost":{"verdict":"holds","void":[],
        "licensed":"On one RTX 3090 at batch 32, c* on Qwen3-VL-Embedding-2B embeds a painting for emotion, style and genre in 4.6 ms (one Qwen3-VL-Embedding-2B pass and three small heads), against 10.1 ms for the Qwen image instructed per aspect (three passes); model time, images decoded beforehand (5.2 and 11.8 ms with decoding).",
        "limit":"The slowest c* pass takes 4.7 ms against the fastest teacher pass at 10.1 ms."}}},
      "counts":{"mmart_n":16384,"semart_n":11159,"semart_artists":2717}
    }''')
    status=c.good_status();actual['status']=status
    cell=dict(sc='SC14',encoder='qwen',aspect='genre',task='ret',data='semart_new',candidate='cstar_qwen',comparator='teacher_genre',
              candidate_value=.3,comparator_value=.2,d=.1,lo=.08,hi=.12,r=.02,outcome='WIN',status='decided',void=[])
    claim={k:cell[k] for k in c.IDENTITY};claim.update(outcome='WIN',status='decided',void=[],claim='better than')
    cost=dict(sc='SC15',encoder='qwen',cstar_ms_max=4.72,teacher_ms_min=10.06,cstar_ms_mean=4.61,teacher_ms_mean=10.14,cstar_ms_e2e=5.24,teacher_ms_e2e=11.76)
    expected=dict(cells=[cell],claims=[claim],cost=[cost],verdicts={'qwen':{'sem':'holds','cost':'holds'}},
                  voids={'qwen':{'sem':[],'cost':[]}},counts=dict(mmart_n=16384,semart_n=11159,semart_artists=2717))
    doc=dict(scope='full',decisions=expected,status=status,cost=[cost],draws_clu=2000,draws_ret=10000,kmeans_threads=8)
    out=scratch/'out';root=scratch/'root';(root/'stage4').mkdir(parents=True)
    c.atomic_json(out/'derived.json',doc)
    return actual,root,out


@pytest.mark.parametrize('name',['Qwen3-VL-Embedding-2B','SigLIP 2','SigLIP 1','CLIP L/14','CLIP B/32','CLIP B/16'])
def test_outcomes_digits_of_encoder_names_are_not_numbers(name,scratch):
    actual,root,out=golden_outcomes_fixture(scratch)
    block=actual['verdicts']['qwen']['semart']
    block['licensed']=f'On 11,159 paintings by 2,717 artists, frozen c* on {name} clusters and retrieves by type.'
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    assert c.check_outcomes(root,out)['pass']
    block['licensed']=f'On 11,159 paintings by 2,717 artists, frozen c* on {name} clusters at 17 clusters.'
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    assert not c.check_outcomes(root,out)['pass']


def test_outcomes_cost_sentence_checks_each_slot(scratch):
    actual,root,out=golden_outcomes_fixture(scratch)
    block=actual['verdicts']['qwen']['cost'];good=block['licensed']
    block['licensed']=good.replace('in 4.6 ms','in 10.1 ms').replace('against 10.1 ms','against 4.6 ms')
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    report=c.check_outcomes(root,out)
    assert not report['pass'] and any('/cost_slot/cstar_ms_mean' in x['name'] and not x['pass'] for x in report['items'])
    block['licensed']='c* takes 4.6 ms against 10.1 ms for the teacher (5.2 and 11.8 ms with decoding).'
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    report=c.check_outcomes(root,out)
    assert not report['pass'] and any('/cost_slot/parsed' in x['name'] and not x['pass'] for x in report['items'])


def test_gate_failed_true_gate_and_ok_status_void_nothing():
    status=c.good_status()
    assert not c.gate_failed(status,'viii',dict(viii=True))
    assert c.gate_failed(status,'viii',dict(viii=False))


@pytest.mark.parametrize('recorded,passes',[('teacher_artwork',True),('teacher_emotion_focus',True),('teacher_photo',False)])
def test_outcomes_affection_teacher_label(recorded,passes,scratch):
    actual,root,out=golden_outcomes_fixture(scratch)
    derived=json.loads((out/'derived.json').read_text())
    cell=derived['decisions']['cells'][0];claim=derived['decisions']['claims'][0]
    for x in (cell,claim):x.update(sc='SC6',aspect='emotion',task='clu',data='affection_test',candidate='photo_qwen',comparator='teacher_emotion_focus')
    c.atomic_json(out/'derived.json',derived)
    for x in (actual['cells'][0],actual['claims'][0]):x.update(sc='SC6',aspect='emotion',task='clu',data='affection_test',candidate='photo_qwen',comparator=recorded)
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    assert c.check_outcomes(root,out)['pass'] is passes


def real_layout_fixture(scratch):
    """The layout stage 4 writes: no status key, and claim rows filed per encoder without an encoder field."""
    actual,root,out=golden_outcomes_fixture(scratch)
    derived=json.loads((out/'derived.json').read_text())
    records=scratch/'records';records.mkdir()
    c.atomic_json(records/'status.json',derived['status'])
    claim=dict(actual['claims'][0]);encoder=claim.pop('encoder')
    claim.update(sentence='c* on Qwen3-VL-Embedding-2B is better than the Qwen image instructed for genre on 11,159 paintings.')
    actual['claims']={encoder:[claim]};actual.pop('status',None)
    return actual,root,out,records,claim


def test_outcomes_real_layout_passes_and_compares_claims(scratch):
    actual,root,out,records,claim=real_layout_fixture(scratch)
    path=root/'stage4/outcomes.json';path.write_text(json.dumps(actual))
    report=c.check_outcomes(root,out,records)
    assert report['pass'] and any(x['name'].endswith('/0/claim') for x in report['items'])
    claim['claim']='level with or better than';path.write_text(json.dumps(actual))
    assert not c.check_outcomes(root,out,records)['pass']
    claim['claim']='better than';claim['sentence']=claim['sentence'].replace('11,159','11,160');path.write_text(json.dumps(actual))
    assert not c.check_outcomes(root,out,records)['pass']
    claim['sentence']=claim['sentence'].replace('11,160','11,159');actual['claims']['qwen']=[];path.write_text(json.dumps(actual))
    assert not c.check_outcomes(root,out,records)['pass']


def test_outcomes_without_status_key_reads_the_committed_record(scratch):
    actual,root,out,records,_=real_layout_fixture(scratch)
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    status=json.loads((records/'status.json').read_text());status['gates']['vi']='failed'
    c.atomic_json(records/'status.json',status)
    assert not c.check_outcomes(root,out,records)['pass']


def test_golden_outcomes_sentences_round_to_printed_decimal(scratch):
    actual,root,out=golden_outcomes_fixture(scratch)
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    assert c.check_outcomes(root,out)['pass']
    actual['verdicts']['qwen']['cost']['limit']='The slowest c* pass takes 4.8 ms against the fastest teacher pass at 10.1 ms.'
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    report=c.check_outcomes(root,out)
    assert not report['pass'] and any('/sentence_number/' in x['name'] and not x['pass'] for x in report['items'])


@pytest.mark.parametrize('fault',['wrong_claim','wrong_void','wrong_cell_void','wrong_verdict_void','absent_claim','absent_claim_field'])
def test_independent_outcomes_claim_and_void_failures(fault,scratch):
    actual,root,out=golden_outcomes_fixture(scratch)
    if fault=='wrong_claim':actual['claims'][0]['claim']='level with or better than'
    elif fault=='wrong_void':actual['claims'][0]['void']=['gates.vi']
    elif fault=='wrong_cell_void':actual['cells'][0]['void']=['gates.vi']
    elif fault=='wrong_verdict_void':actual['verdicts']['qwen']['semart']['void']=['gates.vi']
    elif fault=='absent_claim_field':actual['claims'][0].pop('claim')
    else:actual['claims']=[]
    (root/'stage4/outcomes.json').write_text(json.dumps(actual))
    report=c.check_outcomes(root,out)
    assert not report['pass']
    assert any(not x['pass'] and (x['name'].endswith('/claim') if fault=='wrong_claim' else
                                x['name'].endswith('/void') if fault in ('wrong_void','wrong_cell_void','wrong_verdict_void') else x['name']=='coverage/claims') for x in report['items'])


def test_mmart_count_comes_from_derived_keys(scratch,monkeypatch):
    records=scratch/'records';records.mkdir();(records/'status.json').write_text(json.dumps(c.good_status()))
    monkeypatch.setattr(c,'RECORDS',records)
    monkeypatch.setattr(c.Work,'population',lambda *args:(['derived-a','derived-b','derived-c'],{}))
    row=dict(sc='SC7',encoder='clip_b32',aspect='emotion',task='clu',data='mmart_test',candidate='cstar_clip_b32',comparator='frozen_clip_b32',lo=.1,r=None)
    c.atomic_json(scratch/'sections/smoke:clip_b32:mm:clu.json',dict(name='smoke:clip_b32:mm:clu',complete=True,root=str(c.DEFAULT_ROOT),draws_clu=2000,draws_ret=10000,kmeans_threads=8,source_sha256='fixture',inputs={},result=[row]))
    doc=c.assemble(scratch,False)
    assert doc['mmart_test_keys']==['derived-a','derived-b','derived-c']
    assert doc['decisions']['counts']['mmart_n']==3


def test_assemble_reconciles_measured_gates_with_status(scratch,monkeypatch):
    records=scratch/'records';records.mkdir();(records/'status.json').write_text(json.dumps(c.good_status()))
    monkeypatch.setattr(c,'RECORDS',records)
    for name,result in [('set',dict(n_set=11159,artists_set=2717,gate={'pass':False})),
                        ('cost',dict(cells=all_cost(),gate={'pass':False},table={}))]:
        c.atomic_json(scratch/f'sections/{name}.json',dict(name=name,complete=True,root=str(c.DEFAULT_ROOT),draws_clu=2000,draws_ret=10000,kmeans_threads=8,source_sha256='fixture',inputs={},result=result))
    doc=c.assemble(scratch,False)
    assert doc['derived_gates']=={'vi':False,'viii':False}
    assert all(v['sem']==v['cost']=='inconclusive' for v in doc['decisions']['verdicts'].values())
    assert all(row['status']=='inconclusive' for row in doc['decisions']['cost'])


def test_unchanged_dependency_keeps_original_source_provenance(scratch,monkeypatch):
    work=c.Work(out=scratch);monkeypatch.setattr(c,'source_identity',lambda:'new-consumer')
    doc=dict(name='ranges:clip_b32:aff',complete=True,root=str(c.DEFAULT_ROOT),draws_clu=2000,draws_ret=10000,kmeans_threads=8,source_sha256='original-range-code',inputs={},result={'emotion':{'ret':{'r':.01}}})
    c.atomic_json(scratch/'sections/ranges:clip_b32:aff.json',doc)
    assert work.dep('ranges:clip_b32:aff')==doc['result']
    assert c.read_json(scratch/'sections/ranges:clip_b32:aff.json')['source_sha256']=='original-range-code'
    with pytest.raises(AssertionError,match='source changed'):work.validate(doc)
