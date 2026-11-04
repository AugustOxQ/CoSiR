"""Independent r17 C13: rule + saved inputs, with a compare-only result boundary.

No ns module or current scorer is imported. K-means, weighted metrics, joint
four-metric bootstrap, untold joins, new picks, CRL and decisions are independent.
Frozen retrieval maps/AP and loaders are permitted r10--r15 protocol functions.
MMArt store retrieval is unit(unit(raw)); c* retrieval is unit(float16->float32).
Clustering never renormalizes exports; matched probabilities are normalized once.

Each section is a separate process and atomically writes derived/<section>.json.
Dependencies must already exist; they are never silently computed. Only relevant
prerequisites are required (retrieval uses recorded told ranges, not new K ranges).
At most --workers processes, each with one numerical thread, bootstrap a family;
K-means itself uses eight threads for paintings and one for photos (D20).
All caches bind the input arrays, key order, settings and independent source bytes.
The full derive is intentionally a controller task; `list` estimates slot-minutes.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

HERE = Path('/project/CoSiR/src/test/20261103_new_scores')
TEST = Path('/project/CoSiR/src/test')
BASE = Path('/data/SSD2/cosir_r17/20261103_new_scores')
DEFAULT_ROOT, DEFAULT_OUT, SMOKE_OUT = BASE/'outputs', BASE/'c13', BASE/'c13_smoke'
R11 = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R14 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R15 = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
for folder in ('20261102_mmart_test', '20261102_backbones_sweep'):
    sys.path.insert(0, str(TEST/folder))
import mt_c13 as old
frozen = old.frozen
rt_data, rt_readout, rt_stats = frozen.rt_data, frozen.rt_readout, frozen.rt_stats
ac_select, ac_stats = frozen.ac_select, frozen.ac_stats
import mt_data
import bs_emoset
import rt_crl

ENCODERS = ('qwen', 'siglip2', 'siglip1', 'clip_l14')
ASPECTS = ('emotion', 'style', 'genre')
METRICS = ('ami', 'nmi', 'ari', 'acc')
SETS = ('artelingo', 'mmart', 'affection', 'emoset')
SEEDS = tuple(range(5))
HELD = dict(artelingo='held', mmart='test', affection='test', emoset='test')
K = {'artelingo': dict(zip(ASPECTS, (9,27,10))), 'mmart': dict(zip(ASPECTS, (9,20,10))),
     'affection': {'emotion':9}, 'emoset': {'emotion':8}}
VIEW = dict(qwen='qwen_default', siglip2='siglip2', siglip1='siglip1', clip_l14='clip_l14')
TOL = 1e-6
PAIRS = [(d,e) for d in SETS for e in (('qwen',) if d=='emoset' else ENCODERS)]
PICK_UNITS = [('untold',d,e) for d in SETS[:2] for e in ENCODERS] + [
    ('matched',d,e) for d in SETS[:2] for e in ENCODERS if (d,e)!=('artelingo','qwen')] + [('crl','mmart','clip_l14')]
SECTIONS = ['counts','percept_val'] + [f'ranges:{d}:{e}' for d,e in PAIRS] + [
    f'pick:{a}:{d}:{e}' for a,d,e in PICK_UNITS] + [f'cells:clu:{d}:{e}' for d,e in PAIRS] + [
    f'cells:ret:{d}:{e}' for d in SETS[:2] for e in ENCODERS]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def output_path(path):
    p = Path(path).resolve()
    if not any(p.is_relative_to(root) for root in (DEFAULT_OUT, SMOKE_OUT)):
        raise ValueError(f'output outside allowed C13 roots: {p}')
    return p


def write_json(path, value):
    p = output_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pending = p.with_name(p.name + '.' + uuid.uuid4().hex + '.pending')
    pending.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    pending.replace(p)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norm>0).all()
    return x / norm


def untold_join(heads):
    """Check original key lists before any subset join; E, S, G order is binding."""
    assert len(heads)==3
    keys = list(heads[0][0])
    assert all(list(k)==keys for k,_ in heads), 'untold head key lists differ'
    frozen.join_rows(keys, keys)
    assert all(len(x)==len(keys) for _,x in heads)
    return keys, (np.concatenate([unit(x) for _,x in heads], axis=1) / np.float32(np.sqrt(3))).astype(np.float32)


def contingency(y, c, weights=None, shape=None):
    y,c = np.asarray(y),np.asarray(c)
    assert y.ndim==c.ndim==1 and y.shape==c.shape
    w = np.ones(len(y), dtype=np.int64) if weights is None else np.asarray(weights)
    assert w.shape==y.shape and np.issubdtype(w.dtype,np.integer) and (w>=0).all()
    valid = y>=0
    yy,cc = y[valid].astype(np.int64),c[valid].astype(np.int64)
    assert (cc>=0).all()
    shape = shape or (int(yy.max(initial=-1))+1,int(cc.max(initial=-1))+1)
    assert (yy<shape[0]).all() and (cc<shape[1]).all()
    return np.bincount(yy*shape[1]+cc, weights=w[valid], minlength=shape[0]*shape[1]).reshape(shape).astype(np.int64)


@lru_cache(maxsize=64)
def log_factorials(n):
    return gammaln(np.arange(n+1,dtype=np.float64)+1)


def expected_mi(C,n):
    """Exact hypergeometric expectation, vectorized over cluster margins.

    EMI depends only on the margins. Group equal margins, enumerate the support
    of each hypergeometric pair and sum nij/n * log(n*nij/(ai*bj)). This avoids
    sklearn's scalar inner loop at the large populations of this round.
    """
    a,ac=np.unique(C.sum(axis=1),return_counts=True)
    b,bc=np.unique(C.sum(axis=0),return_counts=True)
    g=log_factorials(n);result=0.
    for ai,multiplicity in zip(a,ac):
        ks=np.arange(1,min(int(ai),int(b.max()))+1,dtype=np.int64)[None,:]
        bj=b[:,None]
        valid=(ks<=bj)&(ks>=np.maximum(1,ai+bj-n))
        lp=(g[ai]+g[n-ai]+g[bj]+g[n-bj]-g[n]-g[ks]-g[ai-ks]
            -g[np.maximum(bj-ks,0)]-g[np.maximum(n-ai-bj+ks,0)])
        lp=np.where(valid,lp,-np.inf)
        term=np.exp(lp)*(ks/n)*(np.log(n)+np.log(ks)-np.log(ai)-np.log(bj))
        result+=multiplicity*float(np.sum(term*bc[:,None]))
    return result


def metrics(C):
    """Independent contingency formulas, including the exact hypergeometric EMI."""
    C = np.asarray(C, dtype=np.int64)
    C = C[C.sum(axis=1)>0][:,C.sum(axis=0)>0]
    n = int(C.sum())
    if n==0:
        return np.array([1.,1.,1.,1.])
    a,b = C.sum(axis=1),C.sum(axis=0)
    i,j = np.nonzero(C)
    v = C[i,j].astype(np.float64)
    mi = max(0.,float(np.sum(v/n * (np.log(v)+np.log(n)-np.log(a[i])-np.log(b[j])))))
    ha = float(-np.sum(a/n*np.log(a/n)))
    hb = float(-np.sum(b/n*np.log(b/n)))
    avg = (ha+hb)/2
    nmi = 1. if len(a)==len(b)==1 else mi/avg
    if len(a)==len(b)==1:
        ami=1.
    else:
        emi=float(expected_mi(C,n))
        den=avg-emi
        eps=np.finfo(np.float64).eps
        den=min(den,-eps) if den<0 else max(den,eps)
        ami=(mi-emi)/den
    # Twice the number of unordered pairs avoids intermediate fractions.
    tp=int(np.sum(C*(C-1)))
    pa,pb=int(np.sum(a*(a-1))),int(np.sum(b*(b-1)))
    total=n*(n-1)
    ari=1. if total==0 or (pa==pb==total) or (pa==pb==0) else (
        (tp - pa*pb/total) / ((pa+pb)/2 - pa*pb/total))
    row,col=linear_sum_assignment(C, maximize=True)
    acc=float(C[row,col].sum()/n)
    return np.array([ami,nmi,ari,acc],dtype=np.float64)


def partitions(x,k,set_name):
    x=np.asarray(x,dtype=np.float32)
    assert np.isfinite(x).all() and np.abs(np.linalg.norm(x,axis=1)-1).max()<1e-3
    with threadpool_limits(8 if set_name in SETS[:2] else 1):
        return np.stack([MiniBatchKMeans(n_clusters=k,n_init=3,batch_size=4096,random_state=s).fit(x).labels_
                         for s in range(3)]).astype(np.int32)


def draws(n,count,seeded_comp=False):
    rng=np.random.default_rng(0)
    for _ in range(count):
        rows=rng.integers(0,n,size=n)
        s=rng.choice(5,size=5,replace=True)
        t=rng.choice(5,size=5,replace=True) if seeded_comp else None
        yield rows,s,t


def arm_metrics(parts,y,weights=None,seeds=None):
    parts=np.asarray(parts)
    p=parts[None] if parts.ndim==2 else parts
    seeds=np.arange(len(p)) if seeds is None else seeds
    values={s: np.mean([metrics(contingency(y,state,weights)) for state in p[s]],axis=0)
            for s in np.unique(seeds)}
    return np.mean([values[s] for s in seeds],axis=0)


_BOOT = None

def _boot_init():
    threadpool_limits(1)


def _boot_chunk(chunk):
    cand,comp,y=_BOOT
    out=[]
    for rows,s,t in chunk:
        w=np.bincount(rows,minlength=len(y))
        out.append(arm_metrics(cand,y,w,s)-arm_metrics(comp,y,w,t))
    return np.asarray(out)


def joint_clustering(cand,comp,y,count,workers=8):
    global _BOOT
    point_c,point_b=arm_metrics(cand,y),arm_metrics(comp,y)
    tasks=draws(len(y),count,np.asarray(comp).ndim==3)
    chunks=[]
    chunk=[]
    for task in tasks:
        chunk.append(task)
        if len(chunk)==8:
            chunks.append(chunk);chunk=[]
    if chunk:chunks.append(chunk)
    _BOOT=(cand,comp,y)
    if workers>1:
        with mp.get_context('fork').Pool(workers,initializer=_boot_init) as pool:
            boot=np.concatenate(pool.map(_boot_chunk,chunks))
    else:
        with threadpool_limits(1):boot=np.concatenate([_boot_chunk(c) for c in chunks])
    _BOOT=None
    bounds=np.percentile(boot,[2.5,97.5],axis=0)
    return {m:dict(cand=float(point_c[i]),comp=float(point_b[i]),d=float(point_c[i]-point_b[i]),
                   lo=float(bounds[0,i]),hi=float(bounds[1,i]),n_draws=count) for i,m in enumerate(METRICS)}


def paired_retrieval(cand,comp,count,apart=False):
    cand,comp=np.asarray(cand),np.asarray(comp)
    assert cand.dtype==comp.dtype==np.float64 and cand.shape[-1]==comp.shape[-1]
    if not apart:
        r=rt_stats.paired(cand,comp,draws=count)
        lo,hi=r['interval']
    else:
        assert cand.shape==comp.shape and cand.shape[0]==5
        n=cand.shape[1]
        boot=[]
        # Same draw order as the plain loop; weight-matrix multiplication bounds memory.
        gen=draws(n,count,True)
        for start in range(0,count,250):
            batch=[next(gen) for _ in range(min(250,count-start))]
            w=np.stack([np.bincount(i,minlength=n) for i,_,_ in batch]).astype(np.float64)
            c=w@cand.T/n;b=w@comp.T/n
            s=np.stack([s for _,s,_ in batch]);t=np.stack([t for _,_,t in batch])
            boot.extend(np.take_along_axis(c,s,1).mean(1)-np.take_along_axis(b,t,1).mean(1))
        lo,hi=np.percentile(boot,[2.5,97.5])
    cv,bv=float(cand.mean()),float(comp.mean())
    return dict(cand=cv,comp=bv,d=cv-bv,lo=float(lo),hi=float(hi),n_draws=count)


def outcome(lo,r):
    assert np.isfinite(lo) and (r is None or np.isfinite(r) and r>=0)
    return 'WIN' if lo>0 else 'LEVEL' if r is not None and lo>-r else 'BELOW'


def select_pick(rows,ranges):
    """Independent r11 score, guard and 0.001 tie rule on seed-mean rows."""
    r0=rt_readout.R0
    assert r0 in rows and set(rows[r0])==set(ASPECTS)
    assert all(np.isfinite(ranges[a]) and ranges[a]>0 for a in ASPECTS)
    assert all(set(row)==set(ASPECTS) for row in rows.values())
    base=rows[r0]
    score={ro:sum((row[a]-base[a])/ranges[a] for a in ASPECTS)/3 for ro,row in rows.items()}
    guard={ro:all(row[a]>=base[a]-ranges[a]-1e-12 for a in ASPECTS) for ro,row in rows.items()}
    best=max(score[ro] for ro in rows if guard[ro])
    tied=[ro for ro in rows if guard[ro] and score[ro]>=best-.001-1e-12]
    rank={'none':0,'center':1,'whiten':2}
    pick=min(tied,key=lambda ro:(ro!=r0,ro.k>0,rank[ro.w],ro.k,-(ro.size or 0)))
    return pick,score,guard


def teacher_bar(d,e,a):
    return 'WIN' if (d=='mmart' and a in ('emotion','style')) or d=='emoset' else 'LEVEL_OR_WIN'


def claim_cells():
    cells=[]
    def add(sc,d,e,a,metric,k,comp,kind='class_count'):
        teacher=comp=='teacher'
        token='ami@20' if kind=='fixed20' else metric
        cells.append(dict(id=f'{sc}/{d}/{e}/{a}/{token}/{k if k is not None else "-"}/{comp}',
                          sc=sc,set=d,encoder=e,aspect=a,task='ret' if metric=='map' else 'clu',
                          metric=metric,k=k,kind=kind,candidate='untold' if sc=='SC7' else 'photo' if sc=='SC4' else 'cstar',
                          comparator=comp,bar=teacher_bar(d,e,a) if teacher else 'WIN'))
    for d,e in PAIRS:
        for a,k in K[d].items():
            for m in METRICS:
                if d in SETS[:2]:
                    for sc,comp in [('SC1','plain'),('SC2','matched'),('SC3','teacher')]:add(sc,d,e,a,m,k,comp)
                    if e=='clip_l14':add('SC5',d,e,a,m,k,'crl')
                else:
                    for comp in ('plain','teacher'):add('SC4',d,e,a,m,k,comp)
            if d in SETS[:2]:
                if e=='clip_l14':
                    add('SC5',d,e,a,'ami',20,'crl','fixed20')
                    add('SC6',d,e,a,'map',None,'crl','map')
                add('SC7',d,e,a,'map',None,'plain','map')
                add('SC8',d,e,a,'map',None,'matched','map')
    assert len(cells)==412 and len({c['id'] for c in cells})==412
    return cells


def range_key(c):
    return '/'.join((c['candidate'],c['set'],c['encoder'],c['aspect'],c['metric']))


def pick_keys(c):
    d,e=c['set'],c['encoder']
    if c['sc']=='SC7':return [f'untold/{d}/{e}']
    if c['sc']=='SC8' and (d,e)!=('artelingo','qwen'):return [f'matched/{d}/{e}']
    if c['sc']=='SC6' and d=='mmart':return ['crl/mmart/clip_l14']
    return []


def voids(c,status):
    """§3.5 scoped gates; a missing range voids only LEVEL and non-WIN teachers."""
    g,s=status.get('gates',{}),status.get('stage1',{})
    why=[]
    def gate(name,sub=None,when=True):
        if when:
            value=g.get(name,{}).get(sub,'missing') if sub is not None else g.get(name,'missing')
            if value!='ok':why.append('gates.'+name+(':'+sub if sub is not None else ''))
    sc,d,e=c['sc'],c['set'],c['encoder']
    clu=c['task']=='clu';cc=c['kind']=='class_count'
    gate('i_a',when=clu and d in SETS[:2])
    gate('i_b',when=sc=='SC2')
    gate('i_c',d,when=sc=='SC4')
    gate('i_d',when=cc)
    gate('ii_scorer',when=not clu)
    gate('ii_crl',when=sc in ('SC5','SC6'))
    gate('ii_percept',when=sc=='SC8')
    gate('iii_metric',c['metric'],when=clu)
    gate('iii_count',d+'/'+c['aspect'],when=cc)
    gate('iv',e,when=(sc=='SC8' and (d,e)!=('artelingo','qwen')) or (sc=='SC2' and d=='mmart'))
    gate('v',e,when=sc=='SC7')
    gate('vi',when=bool(pick_keys(c)))
    for key in pick_keys(c):
        if s.get('picks',{}).get(key,'missing')!='ok':why.append('stage1.picks:'+key)
    if cc and s.get('ranges',{}).get(range_key(c),'missing')!='ok':
        if c.get('outcome')=='LEVEL' or (c['comparator']=='teacher' and c.get('outcome')!='WIN'):
            why.append('stage1.ranges:'+range_key(c))
    return why


def decision(c,status=None):
    word=outcome(c['lo'],c['r'])
    why=voids({**c,'outcome':word},status) if status is not None else []
    teacher=c['comparator']=='teacher'
    decided=not why
    strict=word=='WIN' or (c['bar']=='LEVEL_OR_WIN' and word=='LEVEL')
    meets=word in ('WIN','LEVEL') if teacher else word=='WIN'
    wording=('better than' if c['bar']=='WIN' and word=='WIN' else 'level with or better than') if teacher and meets else ('better than' if meets else None)
    return dict(outcome=word,status='decided' if decided else 'inconclusive',void=why,
                bar_met=strict if decided else None,meets_bar=meets if decided else None,
                wording=wording if decided else None)


@lru_cache(maxsize=1)
def infer_cpu_function():
    """Load only permitted mt_matched.infer_cpu, avoiding its scorer imports.

    The function's source is compiled unchanged. Its globals are the frozen
    trainer dependencies, not mt_matched's orchestration or forbidden scorers.
    Torch is used only by this explicitly permitted CPU inference function.
    """
    import torch
    import rb_matched
    import ct_percept
    path=TEST/'20261102_mmart_test/mt_matched.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='infer_cpu')
    env=dict(np=np,torch=torch,rb_matched=rb_matched,base=ct_percept,Path=Path,threadpool_limits=threadpool_limits)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    torch.set_num_threads(8)
    return env['infer_cpu']


def mapper_path(e,s):
    if e=='qwen':root=TEST/'20261020_claim_test/outputs/stage2/full/percept/core/stage2/k40'
    elif e=='siglip2':root=TEST/'20261025_recipe_backbones/outputs/part2/full/matched/M2'
    else:root=Path('/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/full/matched')/('M8' if e=='siglip1' else 'M7')
    return root/f'P3__s{s}.npz'


@lru_cache(maxsize=16)
def keys(d,split):
    if d=='artelingo':return rt_data.names(split)
    if d=='mmart':return mt_data.keys(split)
    if d=='affection':return frozen.ac_data.keys(split)
    return bs_emoset.keys(split,cut=True)


@lru_cache(maxsize=16)
def labels(d,split):
    ks=keys(d,split)
    if d=='artelingo':return rt_data.labels(split)
    if d=='mmart':return {a:mt_data.labels(ks,a) for a in ASPECTS}
    if d=='affection':return {'emotion':frozen.ac_data.labels(split)}
    names=bs_emoset.labels(ks)
    classes={n:i for i,n in enumerate(frozen.P.EMOSET_CLASSES)}
    return {'emotion':np.asarray([classes[n] for n in names],dtype=np.int64)}


@lru_cache(maxsize=1)
def source_identity():
    tree = ast.parse(Path(__file__).read_text())
    omitted = {'Comparison', 'compare', 'full_scope', 'certify', 'main'}
    tree.body = [node for node in tree.body if getattr(node, 'name', None) not in omitted
                 and not (isinstance(node, ast.If) and '__name__' in ast.unparse(node.test))]
    own = ast.dump(tree, include_attributes=False)
    files = [Path(old.__file__), Path(frozen.__file__), TEST/'20261102_mmart_test/mt_matched.py']
    modules = (rt_data, rt_readout, rt_stats, frozen.rt_metrics, rt_data.rb_encoders, ac_select, ac_stats, mt_data, bs_emoset, rt_crl, frozen.ac_data, frozen.ac_crl, frozen.bs_stores, frozen.rs_vh, old.mp_stores)
    files += [Path(m.__file__) for m in modules]
    return hashlib.sha256((own+''.join(str(p)+sha(p) for p in sorted(set(files)))).encode()).hexdigest()


def recorded_ranges(d,e,task='ret'):
    """D17 and SC5: older frozen records are explicit saved inputs, not targets."""
    if d=='artelingo' and e=='qwen':
        doc=read_json(TEST/'20261029_retrieval/stage0a_ranges.json')['map']
        for a,row in doc.items():
            assert abs(row['r']-np.ptp(list(row['per_seed'].values()) if isinstance(row['per_seed'],dict) else row['per_seed']))<1e-12
        return {a:row['r'] for a,row in doc.items()}
    folder='20261102_mmart_test' if d=='mmart' else '20261102_backbones_sweep'
    block=('ret_ranges' if task=='ret' else 'clu_ranges') if d=='mmart' else ('artret_ranges' if task=='ret' else 'artclu')
    doc=read_json(TEST/folder/'records'/f'stage1_{e}.json')[block]
    for a in ASPECTS:
        vals=doc['per_seed'][a]
        assert abs(doc['ranges'][a]-np.ptp(list(vals.values()) if isinstance(vals,dict) else vals))<1e-12
    return doc['ranges']


def reused_pick(d,e,arm):
    if arm=='crl':assert d=='artelingo';return 'none+k10'
    if d=='mmart':return read_json(TEST/'20261102_mmart_test/records'/f'stage1_{e}.json')['picks']['arms']['frozen' if arm=='plain' else arm]['pick']
    if e=='qwen':
        return read_json(R11/'stage4/picks.json')[{'matched':'percept'}.get(arm,arm)]
    return read_json(TEST/'20261102_backbones_sweep/records'/f'stage1_{e}.json')['artret_picks']['frozen' if arm=='plain' else arm]['pick']


class Work:
    def __init__(self,out,clu=2000,ret=10000,workers=8):
        self.out=output_path(out);self.clu,self.ret,self.workers=clu,ret,workers
        self.inputs={};self.memory={}

    def dependency(self,section):
        path=self.out/'derived'/f'{section}.json'
        if not path.exists():raise FileNotFoundError(f'missing prerequisite {section}; derive it first: {path}')
        doc=read_json(path)
        if not doc.get('complete') or doc.get('source_sha256')!=source_identity():
            raise ValueError(f'stale/incomplete prerequisite {section}; rederive it')
        return doc['result']

    def array(self,name,inputs,fn):
        h=hashlib.sha256(source_identity().encode())
        for x in inputs:
            if isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype.str)).encode());h.update(np.ascontiguousarray(x).tobytes())
            else:h.update(json.dumps(x,sort_keys=True).encode())
        p=self.out/'cache'/f'{name}__{h.hexdigest()}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:return z['value']
        t=time.monotonic();value=np.asarray(fn());assert np.isfinite(value).all()
        p.parent.mkdir(parents=True,exist_ok=True)
        pending=p.with_name(p.stem+'.'+uuid.uuid4().hex+'.pending.npz')
        np.savez_compressed(pending,value=value);pending.replace(p)
        print(f'{name}: {time.monotonic()-t:.1f}s',flush=True)
        return value

    def checked_npz(self,path,key,key_field=None):
        side=read_json(path.with_suffix('.json'))
        digest=sha(path);self.inputs[str(path)]=digest
        assert side.get('sha256')==digest, f'export checksum: {path}'
        with np.load(path,allow_pickle=False) as z:
            x=z[key];ks=z[key_field].tolist() if key_field else None
        assert np.isfinite(x).all()
        if ks is not None:
            frozen.join_rows(ks,ks)
            if 'keys_sha256' in side:assert old.keys_sha(ks)==side['keys_sha256']
        return ks,x

    def cstar_head(self,d,split,e,a,s):
        if d=='artelingo':
            ks=keys(d,split)
            x=rt_data.raw('cstar',split,a,s) if e=='qwen' else frozen.cstar(e,split,a,s)
            return ks,x.astype(np.float32)
        path=R15/'exports'/('val' if split=='val' else 'new')/e/'cstar'/f'{a}__s{s}.npz'
        have,x=self.checked_npz(path,'emb','keys')
        side=read_json(path.with_suffix('.json'))
        assert (side['encoder'],side['aspect'],side['seed'],side['keyset'],side['view']) == (e,a,s,'val' if split=='val' else 'new',VIEW[e])
        assert x.dtype==np.float16 and x.shape==(len(have),2048)
        return have,x

    def vector(self,d,split,e,arm,a='emotion',s=None,retrieval=False):
        ks=keys(d,split)
        if arm=='cstar':
            have,x=self.cstar_head(d,split,e,a,s);x=x[frozen.join_rows(have,ks)].astype(np.float32)
            return unit(x) if retrieval else x
        if arm=='untold':
            have,x=untold_join([self.cstar_head(d,split,e,b,s) for b in ASPECTS])
            return x[frozen.join_rows(have,ks)]
        if arm=='photo':return frozen.exported(d,'photo_'+e,split,s,ks).astype(np.float32)
        if arm=='matched':
            if split=='val' and not (d=='artelingo' and e=='qwen' and retrieval):
                dep=self.dependency('percept_val');p=Path(dep[e][str(s)]['path'])
                with np.load(p,allow_pickle=False) as z:have,x=z['names'].tolist(),z['probs']
                assert sha(p)==dep[e][str(s)]['sha256']
                want=ks if d=='artelingo' else [mt_data.artelingo_name(k) for k in ks]
                x=x[frozen.join_rows(have,want)]
            elif d=='artelingo':
                if e=='qwen':x=rt_data.raw('percept',split,a,s)
                else:
                    p=mapper_path(e,s);p=p.parents[2]/'held'/old.MATCHED[e]/p.name
                    side=read_json(p.with_suffix('.json'))
                    assert side.get('list_sha256',side.get('held_sha256'))==frozen.rs_vh.names_sha(ks)
                    with np.load(p,allow_pickle=False) as z:x=z['prob']
                    assert side.get('array_sha256s',{}).get('prob',frozen.rs_vh.array_sha(x))==frozen.rs_vh.array_sha(x)
            else:
                p=R15/'exports/new'/e/'matched'/f'probs__s{s}.npz'
                have,x=self.checked_npz(p,'probs','keys')
                side=read_json(p.with_suffix('.json'))
                assert (side['encoder'],side['seed'],side['view'])==(e,s,VIEW[e])
                assert x.dtype==np.float32 and x.shape==(len(have),40)
                x=x[frozen.join_rows(have,ks)]
            return unit(x) if retrieval else x
        if arm=='crl':
            img=self.vector(d,split,'clip_l14','plain')
            return frozen.ac_crl.vectors(img,rt_crl.text_array(a)).astype(np.float32)
        view=old.TEACHER_VIEW[a] if arm=='teacher' else VIEW[e]
        if d=='mmart':
            x=mt_data.features(view,ks)
            return unit(x) if retrieval else x
        if d=='artelingo':
            if arm=='teacher':return rt_data.embed('instructed',split,a)
            return rt_data.embed('plain',split,a) if e=='qwen' else frozen.art_image(e,split)
        if arm=='teacher':
            wording = (read_json(Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs/stage1/picks.json'))['affection']['teacher_wording']['clustering']
                       if d=='affection' else read_json(TEST/'20261102_backbones_sweep/records/stage2_emoset.json')['wording']['clustering'])
            assert wording=='artwork', f'photo teacher wording differs: {wording}'
        return unit(frozen.bs_stores.load(d,view,ks))

    def parts(self,d,split,e,arm,a,k,s=None):
        # Shared plain and matched partitions serve all aspects with the same K.
        tag=(d,split,e,arm,a if arm not in ('plain','matched') else '-',k,s)
        if tag not in self.memory:
            x=self.vector(d,split,e,arm,a,s)
            if arm=='matched':x=unit(x)
            self.memory[tag]=self.array('partitions',[keys(d,split),x,k,d],lambda:partitions(x,k,d))
        return self.memory[tag]

    def arm_parts(self,d,split,e,arm,a,k):
        if arm in ('cstar','photo','matched'):
            return np.stack([self.parts(d,split,e,arm,a,k,s) for s in SEEDS])
        return self.parts(d,split,e,arm,a,k)

    def counts(self):
        return {d:{split:{a:int(len(np.unique(y[y>=0]))) for a,y in labels(d,split).items()}
                   for split in ('val',HELD[d])} for d in SETS}

    def percept_val(self):
        result={}
        names=keys('artelingo','val')
        for e in ENCODERS:
            result[e]={};x=self.vector('artelingo','val',e,'plain')
            for s in SEEDS:
                p=mapper_path(e,s)
                prob=self.array('percept_val',[names,x,sha(p),sha(p.with_suffix('.pt')),sha(p.with_suffix('.json'))],lambda:infer_cpu_function()(p,x))
                target=self.out/'percept_val'/e/f'matched__s{s}.npz'
                target.parent.mkdir(parents=True,exist_ok=True)
                pending=target.with_name(target.stem+'.pending.npz')
                np.savez_compressed(pending,names=np.asarray(names),probs=prob);pending.replace(target)
                result[e][str(s)]=dict(path=str(target),sha256=sha(target),shape=list(prob.shape))
        return result

    def ranges(self,d,e):
        out={};arm='cstar' if d in SETS[:2] else 'photo'
        for a,y in labels(d,'val').items():
            k=int(len(np.unique(y[y>=0])));p=self.arm_parts(d,'val',e,arm,a,k)
            values=np.stack([arm_metrics(p[s],y) for s in SEEDS])
            for i,m in enumerate(METRICS):
                out[f'{arm}/{d}/{e}/{a}/{m}']=dict(k=k,n_val=len(y),n_labelled=int((y>=0).sum()),
                    r=float(np.ptp(values[:,i])),per_seed={str(s):float(values[s,i]) for s in SEEDS})
        return out

    def pick(self,arm,d,e):
        self.dependency('percept_val')
        lab=labels(d,'val');ranges=recorded_ranges(d,e);tables=[]
        for s in SEEDS if arm!='crl' else (None,):
            # Aspect-free matrices share identity, reusing each eigendecomposition.
            if arm=='crl':xs={(a,None):self.vector(d,'val',e,arm,a,retrieval=True) for a in ASPECTS}
            else:
                x=self.vector(d,'val',e,arm,s=s,retrieval=True);xs={(a,None):x for a in ASPECTS}
            ros=rt_readout.menu(next(iter(xs.values())).shape[1])
            arrays=[xs[(a,None)] for a in ASPECTS]
            def calc():
                rows=rt_readout.crossfit_all(xs,lab,ros,workers=self.workers)
                return np.asarray([[rows[r][a] for a in ASPECTS] for r in ros])
            tables.append(self.array('crossfit',[keys(d,'val'),*arrays,*[lab[a] for a in ASPECTS]],calc))
        means=np.mean(tables,axis=0)
        rows={r:{a:float(means[i,j]) for j,a in enumerate(ASPECTS)} for i,r in enumerate(ros)}
        pick,scores,guard=select_pick(rows,ranges)
        table={rt_readout.name(r):dict(map=rows[r],score=float(scores[r]),kept=bool(guard[r])) for r in ros}
        return dict(key=f'{arm}/{d}/{e}',pick=rt_readout.name(pick),w=pick.w+str(pick.size or '') if pick.w=='whiten' else pick.w,k=pick.k,
                    table=table,unit=ranges,n_val=len(keys(d,'val')),scores={**table[rt_readout.name(pick)],'r0_map':table['none+k0']['map']})

    def cells_clu(self,d,e):
        ranges=self.dependency(f'ranges:{d}:{e}')
        if d in SETS[:2]:self.dependency('percept_val')
        out=[];cache={}
        for c in claim_cells():
            if (c['set'],c['encoder'],c['task'])!=(d,e,'clu'):continue
            a,k=c['aspect'],c['k'];y=labels(d,HELD[d])[a]
            tag=(a,k,c['candidate'],c['comparator'])
            if tag not in cache:
                cand=self.arm_parts(d,HELD[d],e,c['candidate'],a,k)
                comp=self.arm_parts(d,HELD[d],e,c['comparator'],a,k)
                cache[tag]=joint_clustering(cand,comp,y,self.clu,self.workers)
            stats=cache[tag][c['metric']]
            r=recorded_ranges(d,e,'clu')[a] if c['kind']=='fixed20' else ranges[range_key(c)]['r']
            row={**c,**stats,'r':r,'n_rows':len(y),'n_labelled':int((y>=0).sum())}
            out.append({**row,**decision(row)})
        return out

    def get_pick(self,d,e,arm):
        section=f'pick:{arm}:{d}:{e}'
        if (arm,d,e) in PICK_UNITS:return self.dependency(section)['pick']
        return reused_pick(d,e,arm)

    def ap(self,d,e,arm,a,y):
        pick=self.get_pick(d,e,arm);aps=[]
        for s in SEEDS if arm in ('cstar','untold','matched') else (None,):
            val=self.vector(d,'val',e,arm,a,s,True);test=self.vector(d,HELD[d],e,arm,a,s,True)
            def calc():
                res=ac_stats.arm_retrieval(lambda _:val,lambda _:test,pick,y,workers=self.workers)
                assert np.array_equal(res['rows'],np.flatnonzero(y>=0)),'retrieval dropped/reordered rows'
                return res['ap']
            aps.append(self.array('picked_ap',[keys(d,'val'),keys(d,HELD[d]),val,test,pick,y],calc))
        return (np.stack(aps) if arm in ('cstar','untold','matched') else aps[0]),pick

    def cells_ret(self,d,e):
        self.dependency('percept_val')
        for arm,dd,ee in PICK_UNITS:
            if (dd,ee)==(d,e):self.dependency(f'pick:{arm}:{d}:{e}')
        out=[];cache={}
        for c in claim_cells():
            if (c['set'],c['encoder'],c['task'])!=(d,e,'ret'):continue
            a=c['aspect'];y=labels(d,HELD[d])[a]
            def ap(arm):
                if (arm,a) not in cache:cache[arm,a]=self.ap(d,e,arm,a,y)
                return cache[arm,a]
            cand,cp=ap(c['candidate']);comp,bp=ap(c['comparator'])
            stats=paired_retrieval(cand,comp,self.ret,c['sc']=='SC8')
            row={**c,**stats,'r':recorded_ranges(d,e)[a],'candidate_readout':cp,'comparator_readout':bp,'n_rows':len(y),'n_queries':cand.shape[-1]}
            out.append({**row,**decision(row)})
        return out


def estimate(section):
    if section=='counts':return 1
    if section=='percept_val':return 2
    bits=section.split(':')
    if bits[0]=='ranges':return 7 if bits[1] in SETS[:2] else 2
    if bits[0]=='pick':return 12 if bits[1]=='untold' else 3
    if bits[1]=='ret':return 22 if bits[-1]=='clip_l14' else 18
    return (24 if bits[-1]=='clip_l14' else 18) if bits[2] in SETS[:2] else 9


def derive(sections,out=DEFAULT_OUT,clu=2000,ret=10000,workers=8):
    assert 1<=workers<=8 and 1<=clu<=2000 and 1<=ret<=10000
    assert sections and len(set(sections))==len(sections) and set(sections)<=set(SECTIONS)
    for section in sections:
        w=Work(out,clu,ret,workers);start=time.monotonic();bits=section.split(':')
        print(f'C13 derive {section}',flush=True)
        if section in ('counts','percept_val'):result=getattr(w,section)()
        elif bits[0]=='ranges':result=w.ranges(*bits[1:])
        elif bits[0]=='pick':result=w.pick(*bits[1:])
        else:result=getattr(w,'cells_'+bits[1])(*bits[2:])
        doc=dict(section=section,complete=True,source_sha256=source_identity(),draws_clu=clu,draws_ret=ret,
                 seconds=time.monotonic()-start,inputs=w.inputs,result=result)
        write_json(Path(out)/'derived'/f'{section}.json',doc)
        print(f'C13 done {section} seconds={doc["seconds"]:.1f}',flush=True)


class Comparison:
    def __init__(self):self.items=[]

    def add(self,name,derived,recorded,numeric=False):
        diff=None
        if numeric and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in (derived,recorded)):
            value=abs(float(derived)-float(recorded));ok=bool(np.isfinite(value) and value<=TOL)
            diff=value if np.isfinite(value) else None
        else:ok=type(derived) is type(recorded) and derived==recorded
        self.items.append(dict(name=name,derived=derived,recorded=recorded,abs_diff=diff,**{'pass':ok}))

    def tree(self,name,a,b):
        if isinstance(a,dict):
            self.add(name+'.keys',sorted(a),sorted(b) if isinstance(b,dict) else None)
            for k,v in a.items():self.tree(name+'.'+k,v,b.get(k) if isinstance(b,dict) else None)
        elif isinstance(a,list):
            self.add(name+'.length',len(a),len(b) if isinstance(b,list) else None)
            for i,v in enumerate(a):self.tree(name+f'.{i}',v,b[i] if isinstance(b,list) and len(b)>i else None)
        else:self.add(name,a,b,numeric=isinstance(a,float))

    def report(self):
        sections={}
        for item in self.items:
            stats=sections.setdefault(item['name'].split('/')[0],dict(items=0,failed=0,max_abs=0.))
            stats['items']+=1;stats['failed']+=not item['pass'];stats['max_abs']=max(stats['max_abs'],item['abs_diff'] or 0.)
        return dict(items=self.items,items_compared=len(self.items),failed=sum(not x['pass'] for x in self.items),
                    max_abs=max((x['abs_diff'] or 0. for x in self.items),default=0.),sections=sections)


def full_scope(docs):
    """Coverage is checked against the rule, never inferred from existing outputs."""
    if set(docs)!=set(SECTIONS):return False
    if any(not x.get('complete') or x.get('source_sha256')!=source_identity() for x in docs.values()):return False
    for section,doc in docs.items():
        if section.startswith('cells:'):
            task=section.split(':')[1]
            if doc.get('draws_clu' if task=='clu' else 'draws_ret')!=(2000 if task=='clu' else 10000):return False
    counts=docs['counts']['result']
    if counts!={d:{split:K[d] for split in ('val',HELD[d])} for d in SETS}:return False
    pv=docs['percept_val']['result']
    if set(pv)!=set(ENCODERS) or any(set(pv[e])!=set(map(str,SEEDS)) for e in ENCODERS):return False
    for d,e in PAIRS:
        arm='cstar' if d in SETS[:2] else 'photo'
        wanted={f'{arm}/{d}/{e}/{a}/{m}' for a in K[d] for m in METRICS}
        rows=docs[f'ranges:{d}:{e}']['result']
        if set(rows)!=wanted or any(set(x.get('per_seed',{}))!=set(map(str,SEEDS)) for x in rows.values()):return False
    for arm,d,e in PICK_UNITS:
        row=docs[f'pick:{arm}:{d}:{e}']['result']
        width=6144 if arm=='untold' else 40 if arm=='matched' else 50
        if set(row.get('table',{}))!={rt_readout.name(r) for r in rt_readout.menu(width)}:return False
    cells=[c for s,x in docs.items() if s.startswith('cells:') for c in x['result']]
    if len(cells)!=412 or {c['id'] for c in cells}!={c['id'] for c in claim_cells()}:return False
    return all(c.get('n_draws')==(2000 if c['task']=='clu' else 10000) for c in cells)


def certify(report,records,outcomes_bytes,round_draws):
    reasons=[]
    if report['scope']!='full':reasons.append('scope is a subset or coverage is incomplete')
    if report['failed']:reasons.append('comparison has failures')
    if report.get('skipped'):reasons.append('required comparison evidence was skipped')
    if not round_draws or any(v!=(2000 if task=='clu' else 10000) for task,v in round_draws):reasons.append('round draw counts are not 2000/10000')
    if outcomes_bytes is None:reasons.append('records/outcomes.json is absent')
    # An absence is excused only by a failed gate or stage 1 item (the reason lists it); a bare absence refuses.
    bare=[x['id'] for x in report.get('missing_round_cells',()) if x.get('reason')==['round cell absent']]
    if bare:reasons.append(f'{len(bare)} round cells absent with no failed gate or stage 1 item to excuse them (first: {bare[0]})')
    # Explicit CLI authorization is the sole exception to the output-root rule.
    p=Path(records)/'c13.json'
    def put(doc):
        p.parent.mkdir(parents=True,exist_ok=True)
        pending=p.with_name('c13.json.'+uuid.uuid4().hex+'.pending')
        pending.write_text(json.dumps(doc)+'\n')
        pending.replace(p)
    if reasons:
        # A refusal replaces any older pass, so c13_state cannot still accept it.
        put({'pass':False,'reason':'; '.join(reasons)})
        raise ValueError('refusing --write-record: '+'; '.join(reasons))
    put({'pass':True,'outcomes_sha256':hashlib.sha256(outcomes_bytes).hexdigest(),
         'items':report['items_compared'],'max_abs':report['max_abs']})


def compare(root=DEFAULT_ROOT,records=HERE/'records',out=DEFAULT_OUT,write_record=False):
    """Only this boundary opens r17 records, stage0/stage1 or stage2 result files.

    Layout adaptation changes no numerical values. Missing/inconclusive round
    cells are listed, never failed merely for absence; own full coverage remains
    mandatory. A full run additionally compares all outcome/status rows.
    """
    root,records,out=Path(root),Path(records),output_path(out)
    if write_record and (records/'c13.json').exists():
        # A compare that dies before certify must not leave an older pass standing.
        stale=records/'c13.json';pending=stale.with_name('c13.json.'+uuid.uuid4().hex+'.pending')
        pending.write_text(json.dumps({'pass':False,'reason':'compare did not finish'})+'\n');pending.replace(stale)
    docs={}
    for p in sorted((out/'derived').glob('*.json')):
        doc=read_json(p);section=doc['section']
        if section not in SECTIONS:raise ValueError(f'unknown derived section {section}')
        if section in docs:raise ValueError(f'duplicate section {section}')
        docs[section]=doc
    full=full_scope(docs);scope='full' if full else 'subset'
    result=Comparison();skipped=[];missing_cells=[];round_draws=[];record_layouts={}
    def layout(node):
        if isinstance(node,dict):return {k:layout(v) for k,v in node.items()}
        if isinstance(node,list):return [len(node),layout(node[0]) if node else None]
        return type(node).__name__
    def row_list(node):
        if isinstance(node,dict):
            if 'id' in node and 'sc' in node:yield node
            else:
                for value in node.values():yield from row_list(value)
        elif isinstance(node,list):
            for value in node:yield from row_list(value)
    status=read_json(records/'status.json') if (records/'status.json').exists() else None
    if status is not None:
        expected_groups={'i_c':set(SETS[2:]),'iii_metric':set(METRICS),
                         'iii_count':{d+'/'+a for d in SETS for a in K[d]},
                         'iv':set(ENCODERS),'v':set(ENCODERS)}
        scalar_gates={'i_a','i_b','i_d','ii_scorer','ii_crl','ii_percept','vi'}
        errors=[];g=status.get('gates',{});st=status.get('stage1',{})
        if set(status)!={'gates','stage1'}:errors.append('top-level status keys')
        if set(g)!=scalar_gates|set(expected_groups):errors.append('gate keys')
        if set(st)!={'ranges','picks'}:errors.append('stage1 keys')
        for name,wanted in expected_groups.items():
            if not isinstance(g.get(name),dict) or set(g[name])!=wanted:errors.append('gate group '+name)
        flat=[v for name,v in g.items() if name in scalar_gates]
        flat += [v for group in expected_groups for v in (g.get(group,{}) if isinstance(g.get(group),dict) else {}).values()]
        flat += [v for group in ('ranges','picks') for v in st.get(group,{}).values()]
        if not all(v in ('ok','failed','missing') for v in flat):errors.append('invalid status word')
        used_ranges={range_key(c) for c in claim_cells() if c['kind']=='class_count'}
        used_picks={'/'.join(x) for x in PICK_UNITS}
        if set(st.get('ranges',{}))-used_ranges:errors.append('unused range status')
        if set(st.get('picks',{}))-used_picks:errors.append('unused pick status')
        result.add('status/schema',errors,[])
    outcomes_bytes=(records/'outcomes.json').read_bytes() if (records/'outcomes.json').exists() else None
    outcomes_doc=json.loads(outcomes_bytes) if outcomes_bytes is not None else None
    outcome_list=list(row_list(outcomes_doc)) if outcomes_bytes is not None else []
    outcome_rows={c['id']:c for c in outcome_list}
    ranges=read_json(records/'stage1_ranges.json') if (records/'stage1_ranges.json').exists() else {}
    picks=read_json(records/'stage1_picks.json') if (records/'stage1_picks.json').exists() else {}
    for section,doc in docs.items():
        prefix=section+'/'
        result.add(prefix+'complete',doc.get('complete'),True)
        result.add(prefix+'source_sha256',doc.get('source_sha256'),source_identity())
        value=doc['result']
        if section=='counts':
            # Counts are also compared with the binding spec, and every recorded
            # count row if the gate exposes it. A subset may omit the stage0 gate.
            result.tree(prefix+'binding_K',value,{d:{split:K[d] for split in ('val',HELD[d])} for d in SETS})
            path=root/'stage0/iii.json'
            if path.exists():
                gate=read_json(path)
                record_layouts['counts']=layout(gate)
                countdoc=gate.get('counts_by_split',gate.get('class_counts_by_split'))
                if countdoc is not None:result.tree(prefix+'round_K',value,countdoc)
                else:
                    countdoc=gate.get('counts',gate.get('class_counts'))
                    if isinstance(countdoc,dict):
                        if set(countdoc)=={'painting','photo'}:
                            countdoc={**countdoc['painting'],**countdoc['photo']}
                        for d,splitrows in value.items():
                            for a,k in splitrows[HELD[d]].items():
                                other=countdoc.get(d,{}).get(a) if isinstance(countdoc.get(d),dict) else countdoc.get(d+'/'+a)
                                result.add(prefix+d+'/'+a,k,other)
                    else:result.add(prefix+'round_counts_present',False,True)
            elif full:result.add(prefix+'round_counts_present',False,True)
            else:skipped.append('counts: round stage0/iii.json absent; binding counts checked')
        elif section=='percept_val':
            for e,seeds in value.items():
                for s,entry in seeds.items():
                    p=Path(entry['path']);result.add(prefix+e+'/'+s+'/sha256',sha(p),entry['sha256'])
                    with np.load(p,allow_pickle=False) as z:names,ours=z['names'].tolist(),z['probs']
                    p=root/'percept_val'/e/f'matched__s{s}.npz'
                    if not p.exists():result.add(prefix+e+'/'+s+'/present',False,True);continue
                    with np.load(p,allow_pickle=False) as z:have,theirs=z['names'].tolist(),z['probs']
                    result.add(prefix+e+'/'+s+'/keys',names,have)
                    result.add(prefix+e+'/'+s+'/shape',list(ours.shape),list(theirs.shape))
                    if ours.shape==theirs.shape:
                        diff=float(np.abs(ours.astype(np.float64)-theirs.astype(np.float64)).max())
                        result.add(prefix+e+'/'+s+'/max_probability_difference',diff,0.,True)
        elif section.startswith('ranges:'):
            for key,row in value.items():result.tree(prefix+key,row,ranges.get(key))
        elif section.startswith('pick:'):
            key=value['key'];ref=picks.get(key,{})
            for field in ('pick','w','k','unit','n_val','scores'):result.tree(prefix+field,value[field],ref.get(field))
            relative=ref.get('table_path')
            tablepath=root/relative if isinstance(relative,str) else None
            # Accept an absolute round table path only under the requested root.
            if tablepath is not None and tablepath.resolve().is_relative_to(root.resolve()) and tablepath.exists():
                table=read_json(tablepath)
                result.tree(prefix+'table',value['table'],table.get('table'))
            else:result.add(prefix+'whole_table_present',False,True)
        elif section.startswith('cells:'):
            _,task,d,e=section.split(':');p=root/'stage2/cells'/f'{task}__{d}__{e}.json'
            refs={c['id']:c for c in row_list(read_json(p))} if p.exists() else {}
            result.add(prefix+'derived_ids',sorted(c['id'] for c in value),sorted(c['id'] for c in claim_cells() if (c['task'],c['set'],c['encoder'])==(task,d,e)))
            for c in value:
                name=prefix+c['id'];ref=refs.get(c['id'])
                if ref is None:
                    why=voids(c,status) if status is not None else []
                    missing_cells.append(dict(id=c['id'],reason=why or ['round cell absent'],derived=True))
                    continue
                round_draws.append((task,ref.get('n_draws')))
                for field in ('cand','comp','d','lo','hi','r'):
                    result.add(name+'/'+field,c[field],ref.get(field),True)
                for field in ('id','sc','set','encoder','aspect','task','metric','k','candidate','comparator','n_draws','n_rows'):
                    result.add(name+'/'+field,c[field],ref.get(field))
                if c['id'] in outcome_rows:
                    # The numbers of outcomes.json too, not only its words (a None where a number is due fails).
                    orow=outcome_rows[c['id']]
                    for field in ('cand','comp','d','lo','hi','r'):
                        result.add(name+'/outcomes.'+field,c[field],orow.get(field),True)
                    for field in ('n_draws','n_rows'):
                        result.add(name+'/outcomes.'+field,c[field],orow.get(field))
                    if status is not None:
                        rk=range_key(c) if c['kind']=='class_count' else None
                        if rk is None or status.get('stage1',{}).get('ranges',{}).get(rk,'missing')=='ok':
                            result.add(name+'/outcomes.r_used',c['r'],orow.get('r_used'),True)
                for field in (('candidate_readout','comparator_readout','n_queries') if task=='ret' else ('n_labelled',)):
                    result.add(name+'/'+field,c[field],ref.get(field))
                recorded_word=ref.get('outcome',outcome_rows.get(c['id'],{}).get('outcome'))
                if recorded_word is not None or full:
                    result.add(name+'/outcome',outcome(c['lo'],c['r']),recorded_word)
                elif 'cell outcome words: absent from numerical cell files; records/outcomes.json absent' not in skipped:
                    skipped.append('cell outcome words: absent from numerical cell files; records/outcomes.json absent')
                # Recorded cells may carry decisions, otherwise outcome rows do.
                target=outcome_rows.get(c['id'],ref)
                want=decision(c,status) if status is not None else decision(c)
                for field in ('bar','meets_bar','bar_met','wording'):
                    if field in target or full:
                        if full:result.add(name+'/'+field+'.present',field in target,True)
                        result.add(name+'/'+field,c.get('bar') if field=='bar' else want[field],target.get(field))
                if status is not None and outcomes_bytes is not None:
                    result.add(name+'/status',want['status'],target.get('status'))
                    for field in ('outcome','bar_met','meets_bar','wording'):
                        result.add(name+'/decision.'+field,want[field],target.get(field))
                    if 'void' in target:
                        # Reasons are explanations; scope is tested by row status.
                        result.add(name+'/voided',bool(want['void']),bool(target['void']))
            result.add(prefix+'no_extra_round_cells',sorted(set(refs)-{c['id'] for c in value}),[])
    if not docs:result.add('derive/present',False,True)
    if status is None:
        skipped.append('status: records/status.json absent')
        if full:result.add('status/present',False,True)
    if outcomes_bytes is None:
        skipped.append('outcomes: records/outcomes.json absent')
        if full:result.add('outcomes/present',False,True)
    elif full:
        ours={c['id']:c for s,doc in docs.items() if s.startswith('cells:') for c in doc['result']}
        result.add('outcomes/ids',sorted(ours),sorted(outcome_rows))
        result.add('outcomes/row_count',len(outcome_list),412)
        result.add('outcomes/unique_row_ids',len(outcome_rows),len(outcome_list))
        if status is not None:
            result.add('outcomes/smoke',False,outcomes_doc.get('smoke'))
            result.add('outcomes/embedded_status',status,outcomes_doc.get('status'))
            absent={x['id'] for x in missing_cells}
            expect=[];counts={}
            for cid,c in ours.items():
                want=decision(c,status)
                if cid in absent:want['status']='inconclusive' if want['void'] else 'missing'
                tag='ami@20' if c['kind']=='fixed20' else c['metric']
                n=counts.setdefault(c['sc'],{}).setdefault(tag,dict(cells=0,decided=0,inconclusive=0,missing=0,WIN=0,LEVEL=0,BELOW=0,NOT_WIN=0,meets_bar=0,bar_met=0))
                n['cells']+=1;n[want['status']]+=1
                if want['status']=='decided':
                    ranged=c['kind']!='class_count' or status.get('stage1',{}).get('ranges',{}).get(range_key(c),'missing')=='ok'
                    n['WIN' if want['outcome']=='WIN' else want['outcome'] if ranged else 'NOT_WIN']+=1
                    n['meets_bar']+=int(want['meets_bar']);n['bar_met']+=int(want['bar_met'])
                    if want['meets_bar']:expect.append([c['sc'],c['set'],c['encoder'],c['aspect'],c['comparator'],want['wording'],tag])
            families={}
            for sc,d,e,a,comp,wording,tag in expect:families.setdefault((sc,d,e,a,comp,wording),[]).append(tag)
            wanted=sorted([*k,sorted(v)] for k,v in families.items())
            have=outcomes_doc.get('licensed')
            try:have=sorted([x['sc'],x['set'],x['encoder'],x['aspect'],x['comparator'],x['wording'],sorted(x['metrics'])] for x in have)
            except (TypeError,KeyError):have=None
            result.add('outcomes/licensed',wanted,have)
            result.tree('outcomes/summary',counts,outcomes_doc.get('summary'))
            for cid,c in ours.items():
                target=outcome_rows.get(cid,{})
                want=decision(c,status)
                if cid in {x['id'] for x in missing_cells}:
                    want['status']='inconclusive' if want['void'] else 'missing'
                result.add('outcomes/'+cid+'/status',want['status'],target.get('status'))
    report={**result.report(), 'scope':scope,'subset':sorted(docs) if not full else None,
            'draw_counts':{s:{'clu':v['draws_clu'],'ret':v['draws_ret']} for s,v in docs.items()},
            'missing_round_cells':missing_cells,'skipped':skipped,'record_layouts':record_layouts,
            'outcomes_sha256':hashlib.sha256(outcomes_bytes).hexdigest() if outcomes_bytes is not None else None}
    write_json(out/'compare.json',report)
    print(f'C13 {"FAIL" if report["failed"] else "PASS"} items={report["items_compared"]} failed={report["failed"]} missing={len(missing_cells)} max_abs={report["max_abs"]:.12g} scope={scope}',flush=True)
    if write_record:certify(report,records,outcomes_bytes,round_draws)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    sub.add_parser('list')
    d=sub.add_parser('derive');d.add_argument('--sections',nargs='+',required=True)
    d.add_argument('--out',type=Path,default=DEFAULT_OUT)
    d.add_argument('--draws-clu',type=int,default=2000);d.add_argument('--draws-ret',type=int,default=10000)
    d.add_argument('--workers',type=int,default=8)
    c=sub.add_parser('compare');c.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    c.add_argument('--records',type=Path,default=HERE/'records');c.add_argument('--out',type=Path,default=DEFAULT_OUT)
    c.add_argument('--write-record',action='store_true')
    args=parser.parse_args(argv)
    try:
        with threadpool_limits(8):
            if args.cmd=='list':
                # Queue metadata only: calibrated after the subset timing.
                # The numerical fingerprint excludes this CLI function.
                for s in SECTIONS:
                    minutes=estimate(s)
                    if s=='percept_val':minutes=1
                    elif s.startswith('pick:untold:'):minutes=24
                    elif s.startswith('pick:matched:'):minutes=7
                    elif s.startswith('pick:crl:'):minutes=2
                    elif s.startswith('cells:ret:'):
                        minutes=(14 if s.split(':')[2]=='artelingo' else 20)+(3 if s.endswith(':clip_l14') else 0)
                    elif s.startswith('ranges:'):minutes=5 if s.split(':')[1] in SETS[:2] else 1
                    elif s.startswith('cells:clu:') and s.split(':')[2] in SETS[2:]:minutes=3
                    print(f'{s} --est {minutes}m')
            elif args.cmd=='derive':derive(args.sections,args.out,args.draws_clu,args.draws_ret,args.workers)
            else:return int(bool(compare(args.root,args.records,args.out,args.write_record)['failed']))
    except (ValueError,FileNotFoundError) as exc:
        print(f'C13 ERROR: {exc}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
