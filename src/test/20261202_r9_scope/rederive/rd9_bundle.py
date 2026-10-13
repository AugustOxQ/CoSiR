"""Prepare pinned backbone features, head posteriors and A3 codes without the chain."""
import hashlib
import json
from pathlib import Path
import numpy as np
from rd9_inputs import HERE,MAIN,T,sha_bytes,sha_file
from rd9_scores import unit32,cosine,baseline,reader_state,reader_cells,rca_fusion,metrics
from src.eval.aspect_quick_checks import centered_term,uniform_probe_scores
from src.eval.aspect_scorers import EvalInputs

GROUPS=('affect','affect_km','image','caption')


def split(inputs,data):
    z=inputs.npz('prepare',['groups','split_train','scorer_train','selection'])
    n=len(data.sample_ids)
    assert n==308723 and len(np.unique(data.sample_ids))==n
    assert z['groups'].shape==(n,) and len(z['scorer_train'])==183694 and len(z['selection'])==32413
    st,sel=z['scorer_train'],z['selection']
    assert len(np.intersect1d(st,sel))==0
    assert np.array_equal(np.sort(np.concatenate([st,sel])),np.sort(z['split_train']))
    return z


def head_draw_hash(rows):
    # Arm L's provenance hashes the draw as a sorted set; fitting preserves draw order.
    return sha_bytes(np.sort(np.asarray(rows,np.int64)))


def fit_heads(inputs,data,sp):
    from sklearn.linear_model import LogisticRegression
    st,sel=sp['scorer_train'],sp['selection']
    p=inputs.npz('partitions',['affect','image','caption','local_groups'])
    assert np.array_equal(p['local_groups'],np.unique(sp['groups'][st],return_inverse=True)[1])
    pl=inputs.npz('partition_L',['partition_L'])['partition_L']
    labels={'affect':pl,'affect_km':p['affect'],'image':p['image'],'caption':p['caption']}
    draw=np.random.default_rng(0).choice(st,60000,replace=False)
    identity=inputs.json('told_oracle',['arms.L.head.n_classes','arms.L.head.draw_rows_sha256'])['arms']['L']['head']
    assert identity['n_classes']==41 and head_draw_hash(draw)==identity['draw_rows_sha256']
    positions={int(row):j for j,row in enumerate(st)}
    local=np.array([positions[int(row)] for row in draw])
    # D2 uses one deterministic draw for every standard head.
    models,post,hashes={},{},{}
    for g in GROUPS:
        assert labels[g].shape==(len(st),)
        models[g],post[g],hashes[g]={},{},{}
        for m,features in (('img',data.img_features),('txt',data.txt_features)):
            model=LogisticRegression(C=1,max_iter=300).fit(unit32(features[draw]),labels[g][local])
            assert np.array_equal(model.classes_,np.arange(len(model.classes_)))
            models[g][m]=model
            post[g][m]=model.predict_proba(unit32(features[sel])).astype(np.float32)
            digest=hashlib.sha256(np.ascontiguousarray(model.coef_).tobytes())
            digest.update(np.ascontiguousarray(model.intercept_).tobytes())
            hashes[g][m]=digest.hexdigest()
    pinned=inputs.npz('n6_posteriors',['selection',*[f'{g}__{m}' for g in ('affect','image','caption') for m in ('img','txt')]])
    assert np.array_equal(pinned['selection'],sel)
    for g,pinned_g in (('affect_km','affect'),('image','image'),('caption','caption')):
        for m in ('img','txt'):
            assert np.array_equal(post[g][m],pinned[f'{pinned_g}__{m}']) and post[g][m].dtype==np.float32, (g,m,'posterior identity')
    return models,post,hashes


def encoded(inputs,data,rows):
    from src.train.train_factors import load_factor_checkpoint,encode_rows
    record=inputs.json('picked',['run','checkpoint_sha256'])
    checkpoint=inputs.path('A3_seed42',['state_dict','config','feature_dim'])
    assert record['run']=='A3' and record['checkpoint_sha256']==sha_file(checkpoint)
    model,_=load_factor_checkpoint(checkpoint,device='cpu')
    return encode_rows(model,data.img_features,data.txt_features,rows=rows,device='cpu')


def masked(array,rows,n):
    array=np.asarray(array,np.float32)
    assert len(array)==len(rows) and np.isfinite(array).all()
    out=np.full((n,*array.shape[1:]),np.nan,np.float32)
    out[rows]=array
    mask=np.zeros(n,bool);mask[rows]=True
    assert np.isfinite(out[mask]).all() and np.isnan(out[~mask]).all()
    return out


def prepare_arrays(data,rows,post,codes):
    n=len(data.sample_ids)
    assert len(np.unique(rows))==len(rows)
    result={'rows':rows,'sample_ids':data.sample_ids,'img':masked(data.img_features[rows],rows,n),
            'txt':masked(data.txt_features[rows],rows,n),
            'img_codes':masked(codes[0],rows,n),'txt_codes':masked(codes[1],rows,n)}
    result.update({f'{g}__{m}':masked(post[g][m],rows,n) for g in GROUPS for m in ('img','txt')})
    return result


def score_block(bundle,ep,readers,basis,frozen=None,taus=None):
    from src.eval.pair_metric_baselines import rca_term
    from types import SimpleNamespace
    assert np.isin(ep.rows(),bundle['rows']).all()
    n=len(ep.anchor)
    parity=np.arange(n)%2
    img,txt=unit32(bundle['img']),unit32(bundle['txt'])
    cos=cosine(img,txt,ep)
    inputs=EvalInputs(img,txt,bundle['img_codes'],bundle['txt_codes'])
    tu=centered_term(inputs,ep,uniform=True)
    P={g:{m:bundle[f'{g}__{m}'] for m in ('img','txt')} for g in GROUPS}
    scores={'cosine':cos}
    picks={}
    for name,groups in (('B',('affect_km','image','caption')),('B0',('affect','image','caption'))):
        tc=uniform_probe_scores(P,ep,groups)
        scores[name],picks[name]=baseline(cos,tu,tc,parity,None if frozen is None else frozen[name])
    gated,cf,taus,extra=reader_state(P,ep,readers,taus)
    reader_scores,cell_picks=reader_cells(scores['B'],gated,cf,parity,frozen)
    scores.update(reader_scores)
    term=rca_term(SimpleNamespace(img=img,txt=txt),ep,basis)
    scores['rca'],rca_picks=rca_fusion(cos,term,parity,None if frozen is None else frozen['rca_lambda'])
    picks.update(cell_picks)
    picks.update(rca_lambda=rca_picks,taus=taus.tolist())
    arrays={f'{name}__{field}':v for name,score in scores.items() for field,v in metrics(score).items()}
    arrays.update(extra)
    arrays['parity']=parity
    chosen_gate=[]
    for h,cell in enumerate(picks['aff']):
        selected=parity==1-h
        t=cell//56
        chosen_gate.extend([extra[f'aff_gate__{c}'][t,selected] for c in ('a','b')])
    picks['gate_share']=float(np.concatenate(chosen_gate).astype(np.float64).mean())
    return arrays,picks


def fit_basis(inputs,data,sp):
    from src.eval.pair_metric_baselines import fit_pca_basis
    rows=np.random.default_rng(0).choice(sp['scorer_train'],60000,replace=False)
    record=inputs.json('baselines_seed42',['fit_rows_sha256'])
    assert sha_bytes(np.asarray(rows,np.int64))==record['fit_rows_sha256']
    return fit_pca_basis(unit32(data.img_features[rows]),unit32(data.txt_features[rows]))
