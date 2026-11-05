"""Independent r18 C13. Derive reads rules/inputs; finish/compare own the result boundary.

Run `list` for dependency order and slot-minute estimates. Each derive section is
one process; bootstrap/retrieval use at most eight one-thread workers. K-means
uses eight threads. Shared partition caches bind source, keys and input bytes.
Only finish opens status; only compare opens current-round numeric results.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from threadpoolctl import threadpool_limits

ROUND = Path('/project/CoSiR/.claude/worktrees/r18-int/src/test/20261103_outside_methods')
BASE = Path('/data/SSD2/cosir_r18/20261103_outside_methods')
DEFAULT_ROOT = BASE/'outputs'
DEFAULT_OUT = DEFAULT_ROOT/'c13'
SMOKE_OUT = BASE/'c13_smoke'
R17 = ROUND.parent/'20261103_new_scores'
PARITY = Path('/data/SSD2/cosir_r18/ext/parity')
PARITY_SHA256 = {'inputs': '7b10bf69d6fe0358b7b00d98ef974a2d486a73a74f4faa384cc75dd219e4876c',
                 'clay': '3bd542a01e9898dceb4f198c99a4240fc87f81693f089f91c6d0554104c8771a'}
TEXT_A = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs/crl')
TEXT_B = DEFAULT_ROOT/'textset_b'
TPIPS = {d: Path('/data/SSD2/pre_extract')/folder/'tpips_8b'
         for d,folder in [('artelingo','artelingo_full'),('mmart','mmart')]}
SETS = ('artelingo', 'mmart')
ENCODERS = ('qwen','siglip2','siglip1','clip_l14')
ASPECTS = ('emotion','style','genre')
METRICS = ('ami','nmi','ari','acc')
HELD = {'artelingo':'held','mmart':'test'}
K = {'artelingo':dict(zip(ASPECTS,(9,27,10))), 'mmart':dict(zip(ASPECTS,(9,20,10)))}
TOL = 1e-6
SC = {'SC1':('clu',('qwen',),'cstar','tpips'),
      'SC2':('ret',('qwen',),'cstar','tpips'),
      'SC3':('clu',('qwen','clip_l14'),'cstar','clay'),
      'SC4':('ret',('qwen','clip_l14'),'cstar','clay'),
      'SC5':('clu',ENCODERS,'cstar_turtle','plain_turtle'),
      'SC6':('clu',ENCODERS,'cstar','plain_turtle')}
_spec = importlib.util.spec_from_file_location('r17_independent_c13', R17/'ns_c13.py')
ns = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ns
_spec.loader.exec_module(ns)
rt = ns.rt_readout
keys, labels = ns.keys, ns.labels


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
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
    pending = p.with_name(p.name+'.'+uuid.uuid4().hex+'.pending')
    pending.write_text(json.dumps(value, indent=1, allow_nan=False)+'\n')
    pending.replace(p)


@lru_cache(maxsize=1)
def source_identity():
    # Compare layout fixes and runtime estimates cannot invalidate derived arrays.
    own = Path(__file__).read_text().split('\n# RESULT BOUNDARY')[0]
    return hashlib.sha256((own+ns.source_identity()+sha(Path(ns.__file__))).encode()).hexdigest()


def unit64(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def log_map(x, mu):
    cos = np.clip(x @ mu, -1+1e-7, 1-1e-7)
    tangent = x - (x @ mu)[:, None]*mu
    return tangent * (np.arccos(cos)/np.maximum(np.linalg.norm(tangent, axis=1), 1e-9))[:, None]


def reflection(x, v):
    vv = float(v @ v)
    return x.copy() if vv < 1e-12 else x - (2/vv)*(x @ v)[..., None]*v


def align(x, image_mean, mu):
    midpoint = unit64((image_mean+mu)/2)
    return reflection(reflection(x, image_mean-midpoint), midpoint-mu)


def clay_subspace(text, rank=None):
    t = np.asarray(text)
    if t.dtype != np.float32 or t.ndim != 2 or not np.isfinite(t).all() or len(t)<2:
        raise ValueError('CLAY texts must be finite float32 rows, at least two')
    rank = min(50,len(t)-1) if rank is None else rank
    if not 1 <= rank <= min(t.shape[1],len(t)-1):
        raise ValueError('invalid CLAY rank')
    t = t.astype(np.float64)
    mu = unit64(t.mean(axis=0))
    tangent = log_map(t, mu)
    tangent = tangent - tangent.mean(axis=0)
    _, _, vt = np.linalg.svd(tangent, full_matrices=False)
    v = vt[:rank]
    return (v.T @ v).astype(np.float32), mu.astype(np.float32)


def clay_condition(images, projector, mean):
    x = np.asarray(images, dtype=np.float64)
    p, mu = np.asarray(projector, dtype=np.float64), np.asarray(mean, dtype=np.float64)
    image_mean = unit64(x.mean(axis=0))
    z = log_map(align(x, image_mean, mu), mu) @ p
    return unit64(z).astype(np.float32)


def text_inputs():
    doc = read_json(TEXT_B/'lists.json')
    rows = doc.get('aspects', doc)
    available = all(rows.get(a, {}).get('fallback', True) is False and (TEXT_B/f'{a}_text.npy').is_file()
                    for a in ASPECTS)
    result = {}
    for ts, root in [('A',TEXT_A),('B',TEXT_B)]:
        if ts=='B' and not available:
            continue
        result[ts] = {a:np.load(root/f'{a}_text.npy', allow_pickle=False) for a in ASPECTS}
    return result


def tpips_join(d, split, aspect, wanted, roots=None, name_of=None):
    roots = TPIPS if roots is None else roots
    name_of = ns.mt_data.artelingo_name if name_of is None else name_of
    store_set = 'artelingo' if d=='mmart' and split=='val' else d
    want = [name_of(k) for k in wanted] if d=='mmart' and split=='val' else list(wanted)
    p = Path(roots[store_set])/aspect
    have = read_json(p/'keys.json')
    if not isinstance(have,list) or len(set(have)) != len(have) or len(set(want)) != len(want):
        raise ValueError('TPIPS duplicate/invalid keys')
    loc = {k:i for i,k in enumerate(have)}
    missing = set(want)-loc.keys()
    if missing:
        raise ValueError(f'TPIPS missing {len(missing)} requested keys')
    x = np.load(p/'emb.npy', mmap_mode='r', allow_pickle=False)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[0]!=len(have):
        raise ValueError('TPIPS store shape/dtype')
    value = np.asarray(x[[loc[k] for k in want]])
    if not np.isfinite(value).all():
        raise ValueError('nonfinite TPIPS rows')
    return value


def turtle_parts(d, e, arm, k, wanted, aspect=None, root=DEFAULT_ROOT):
    if arm not in ('plain','cstar'):
        raise ValueError('TURTLE arm must be plain or cstar')
    if arm=='cstar' and aspect not in ASPECTS:
        raise ValueError('TURTLE cstar needs an aspect')
    if k not in ({9,10,20,27,5,8,16,40} if arm=='plain' else {K[d][aspect],20}):
        raise ValueError('TURTLE K outside protocol')
    result = []
    for index in range(3 if arm=='plain' else 5):
        start, seed = (index,None) if arm=='plain' else (0,index)
        stem = f'k{k}__start{start}' if arm=='plain' else f'{aspect}__s{seed}__k{k}'
        path = Path(root)/'parts/turtle'/d/e/arm/(stem+'.npz')
        side = read_json(path.with_suffix('.json'))
        expected = dict(method='turtle',set=d,split=HELD[d],enc=e,arm=arm,
                        aspect=None if arm=='plain' else aspect,K=k,start=start,seed=seed,n=len(wanted))
        if any(side.get(key)!=value for key,value in expected.items()):
            raise ValueError(f'TURTLE sidecar describes another fit: {path}')
        if side.get('sha256') != sha(path):
            raise ValueError(f'TURTLE sha256 mismatch: {path}')
        with np.load(path, allow_pickle=False) as z:
            have, part = z['keys'].tolist(), z['labels']
        if have!=list(wanted) or len(set(have))!=len(have):
            raise ValueError(f'TURTLE keys/order mismatch: {path}')
        if part.dtype!=np.int32 or part.shape!=(len(wanted),) or np.any(part<0) or np.any(part>=k):
            raise ValueError(f'TURTLE invalid labels: {path}')
        result.append(part)
    p = np.stack(result)
    return p if arm=='plain' else p[:,None,:]


def repeat_turtle(p):
    if p.ndim!=3 or p.shape[:2]!=(5,1):
        raise ValueError('cstar TURTLE must have shape [5,1,n]')
    return np.repeat(p,3,axis=1)


def cell_specs():
    cells = []
    for sc,(task,encs,candidate,comparator) in SC.items():
        for d in SETS:
            for e in encs:
                for a in ASPECTS:
                    pairs = [(None,('map',))] if task=='ret' else [(K[d][a],METRICS)]
                    if task=='clu' and K[d][a]!=20:
                        pairs.append((20,('ami',)))
                    for k, metrics in pairs:
                        for metric in metrics:
                            cells.append(dict(id=f'{sc}/{d}/{e}/{a}/{metric}/{k if k is not None else "-"}/{comparator}',
                                              sc=sc,set=d,encoder=e,aspect=a,task=task,metric=metric,k=k,
                                              candidate=candidate,comparator=comparator))
    return cells


def range_key(c):
    key = f"{c['set']}/{c['encoder']}/{c['aspect']}/{c['metric']}"
    return key if c['task']=='ret' else key+f"/{c['k']}"


def range_source(c):
    d,e,a,m,k = (c[v] for v in ('set','encoder','aspect','metric','k'))
    if m=='map':
        return 'r11/map' if (d,e)==('artelingo','qwen') else 'r15/ret_ranges' if d=='mmart' else 'r14/artret_ranges'
    if k==20 and m=='ami':
        return 'r15/clu_ranges' if d=='mmart' else 'computed/qwen_ami20' if e=='qwen' else 'r14/artclu'
    return f'r17/cstar/{d}/{e}/{a}/{m}'


def choose_textset(scores, task):
    if task not in ('clu','ret'):
        raise ValueError('text-set task')
    margin = .001 if task=='clu' else .1
    return 'B' if 'B' in scores and scores['B']-scores['A'] > margin+rt.SLACK else 'A'


def label(lo, r):
    if not np.isfinite(lo) or r is not None and (not np.isfinite(r) or r<0):
        raise ValueError('nonfinite interval or invalid range')
    return 'WIN' if lo>0 else 'inconclusive' if r is None else 'LEVEL' if lo>-r else 'BELOW'


def class_count_cell(c):
    return c['task']=='clu' and c['k']==K[c['set']][c['aspect']]


def decide(c, status, previous):
    sc,d,a,m = (c[k] for k in ('sc','set','aspect','metric'))
    why = []
    gates = {'i':('SC3','SC4'),'ii':('SC5','SC6'),'iii_a':('SC1','SC3','SC5','SC6'),
             'iii_b':('SC5','SC6'),'iii_c':('SC2','SC4'),'iv':('SC1','SC2')}
    for gate, families in gates.items():
        if sc in families and status.get('gates',{}).get(gate)!='ok':
            why.append('r18.gates.'+gate)
    need = []
    if sc=='SC2':need = [f'tpips/{d}']
    if sc=='SC3':need = [f'clay_textset/clu/{d}']
    if sc=='SC4':need = [f'clay/{d}',f'clay_textset/ret/{d}']
    stage1 = status.get('stage1',{})
    for key in need:
        if stage1.get(key)!='ok':why.append('r18.stage1.'+key)
    g = previous.get('gates',{})
    need = [('ii_scorer',None),('vi',None)] if c['task']=='ret' else [('i_a',None),('iii_metric',m)]
    if class_count_cell(c):need += [('i_d',None),('iii_count',d+'/'+a)]
    for gate, sub in need:
        value = g.get(gate) if sub is None else g.get(gate,{}).get(sub)
        if value!='ok':why.append('r17.gates.'+gate+(':'+sub if sub else ''))
    rk = range_key(c)
    available = stage1.get(rk)=='ok'
    if range_source(c).startswith('r17/'):
        oldkey = '/'.join(('cstar',d,c['encoder'],a,m))
        available &= previous.get('stage1',{}).get('ranges',{}).get(oldkey)=='ok'
    word = label(c['lo'],c['r'] if available else None)
    return {**c,'label':None if why else word,'void':why or None,'r_used':c['r'] if available else None}


UNITS = list(dict.fromkeys((c['sc'],c['set'],c['encoder']) for c in cell_specs()))
SECTIONS = ['counts','ranges'] + [f'clay:{d}:{split}' for d in SETS for split in ('val',HELD[d])] + [
    f'pick:{arm}:{d}' for d in SETS for arm in ('tpips','clay')] + [f'textset:clu:{d}' for d in SETS] + [
    f'parts:cstar:{d}:{e}' for d in SETS for e in ENCODERS] + [
    f'parts:{arm}:{d}' for d in SETS for arm in ('tpips','clay')] + [f'cells:{sc}:{d}:{e}' for sc,d,e in UNITS]


def dependencies(section):
    b = section.split(':')
    if section=='ranges':return ['counts']
    if b[0] in ('clay',) or section=='counts':return []
    if b[0]=='pick':return ['ranges']+([f'clay:{b[2]}:val'] if b[1]=='clay' else [])
    if b[0]=='textset':return [f'clay:{b[2]}:val']
    if b[:2]==['parts','cstar']:return ['counts']
    if b[0]=='parts':
        return ['counts']+([f'clay:{b[2]}:{HELD[b[2]]}',f'textset:clu:{b[2]}'] if b[1]=='clay' else [])
    if b[0]=='cells':
        _,sc,d,e=b
        if SC[sc][0]=='ret':
            arm=SC[sc][3]
            return ['ranges',f'pick:{arm}:{d}']+([f'clay:{d}:{HELD[d]}'] if arm=='clay' else [])
        result=['counts','ranges']
        if sc!='SC5':result.append(f'parts:cstar:{d}:{e}')
        if sc in ('SC1','SC3'):result.append(f'parts:{SC[sc][3]}:{d}')
        return result
    raise ValueError(f'unknown section {section}')


class Work(ns.Work):
    def __init__(self,out,clu=2000,ret=10000,workers=8):
        self.out=output_path(out)
        self.clu,self.ret,self.workers=clu,ret,workers
        self.inputs,self.memory,self.deps={}, {}, {}

    def dependency(self,section):
        if section in self.deps:return self.deps[section]
        path=self.out/'derived'/f'{section}.json'
        if not path.is_file():raise FileNotFoundError(f'missing prerequisite {section}; derive it first: {path}')
        doc=read_json(path)
        if doc.get('section')!=section or not doc.get('complete') or doc.get('source_sha256')!=source_identity():
            raise ValueError(f'stale/incomplete prerequisite {section}; rederive it')
        self.inputs[str(path)]=sha(path)
        self.deps[section]=doc['result']
        return doc['result']

    def stored(self,name,inputs,fn):
        h=hashlib.sha256(source_identity().encode())
        for x in inputs:
            if isinstance(x,np.ndarray):
                h.update(str((x.shape,x.dtype.str)).encode());h.update(np.ascontiguousarray(x).tobytes())
            else:h.update(json.dumps(x,sort_keys=True).encode())
        p=self.out/'cache'/f'{name}__{h.hexdigest()}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:value=z['value']
        else:
            value=np.asarray(fn())
            if not np.isfinite(value).all():raise ValueError('nonfinite cache array')
            p.parent.mkdir(parents=True,exist_ok=True)
            pending=p.with_name(p.stem+'.'+uuid.uuid4().hex+'.pending.npz')
            np.savez_compressed(pending,value=value);pending.replace(p)
        return value,dict(path=str(p),sha256=sha(p),shape=list(value.shape),dtype=value.dtype.str)

    def array(self,name,inputs,fn):
        return self.stored(name,inputs,fn)[0]

    def artifact(self,entry):
        path=output_path(entry['path'])
        if sha(path)!=entry['sha256']:raise ValueError(f'changed C13 artifact: {path}')
        with np.load(path,allow_pickle=False) as z:x=z['value']
        if list(x.shape)!=entry['shape'] or x.dtype.str!=entry['dtype']:raise ValueError('artifact shape/dtype')
        return x

    def vector(self,d,split,e,arm,a='emotion',s=None,retrieval=False):
        if arm=='tpips':
            tag=('tpips',d,split,a)
            if tag not in self.memory:self.memory[tag]=tpips_join(d,split,a,keys(d,split))
            return self.memory[tag]
        if arm=='clay':
            ts=self.dependency(f'pick:clay:{d}')['text_set']
            return self.clay_vector(d,split,ts,a)
        return super().vector(d,split,e,arm,a,s,retrieval)

    def clay_vector(self,d,split,ts,a):
        tag=('clay',d,split,ts,a)
        if tag not in self.memory:
            entry=self.dependency(f'clay:{d}:{split}')['sets'][ts]['arrays'][a]
            self.memory[tag]=self.artifact(entry)
        return self.memory[tag]

    def counts(self):
        result={d:{split:{a:int(len(np.unique(y[y>=0]))) for a,y in labels(d,split).items()}
                   for split in ('val',HELD[d])} for d in SETS}
        if result!={d:{s:K[d] for s in ('val',HELD[d])} for d in SETS}:raise ValueError('label counts disagree with rule')
        return result

    def ranges(self):
        self.dependency('counts')
        p=R17/'records/stage1_ranges.json'
        r17=read_json(p) if p.exists() else {}
        out={};computed={}
        yall=labels('artelingo','val')
        for a in ASPECTS:
            y=yall[a];values=[]
            for s in range(5):
                x=self.vector('artelingo','val','qwen','cstar',a,s)
                parts=self.array('qwen_val20',[keys('artelingo','val'),x,20,8],lambda:ns.partitions(x,20,'artelingo'))
                values.append(float(ns.arm_metrics(parts,y)[0]))
            computed[a]=dict(r=float(np.ptp(values)),per_seed=values)
        for c in cell_specs():
            rk=range_key(c)
            if rk in out:continue
            src=range_source(c);d,e,a,m=(c[x] for x in ('set','encoder','aspect','metric'))
            extra={}
            if src=='computed/qwen_ami20':
                row=computed[a];r=row['r'];extra=dict(per_seed=row['per_seed'])
            elif src.startswith('r17/'):
                r=r17.get(src[4:],{}).get('r')
            else:
                r=ns.recorded_ranges(d,e,'ret' if m=='map' else 'clu')[a]
            out[rk]=dict(r=r,source=src,**extra)
        return out

    def clay(self,d,split):
        image=self.vector(d,split,'clip_l14','plain')
        texts=text_inputs();out={}
        for ts,rows in texts.items():
            arrays={};ranks={};text_hashes={}
            for a,t in rows.items():
                rank=min(50,len(t)-1);ranks[a]=rank
                text_hashes[a]=hashlib.sha256(t.tobytes()).hexdigest()
                def calc():
                    p,mu=clay_subspace(t,rank)
                    return clay_condition(image,p,mu)
                _,arrays[a]=self.stored('clay',[keys(d,split),image,t,rank],calc)
            out[ts]=dict(arrays=arrays,ranks=ranks,text_sha256=text_hashes)
        return dict(set=d,split=split,n=len(image),sets=out,b_available='B' in texts)

    def pick_table(self,d,arm,ts=None):
        lab=labels(d,'val');encoder='qwen' if arm=='tpips' else 'clip_l14'
        ranges=ns.recorded_ranges(d,encoder)
        xs={(a,None):self.vector(d,'val',encoder,'tpips',a) if arm=='tpips' else self.clay_vector(d,'val',ts,a)
            for a in ASPECTS}
        width=4096 if arm=='tpips' else min(self.dependency(f'clay:{d}:val')['sets'][ts]['ranks'].values())
        ros=rt.menu(width)
        def calc():
            rows=rt.crossfit_all(xs,lab,ros,workers=self.workers)
            return np.asarray([[rows[r][a] for a in ASPECTS] for r in ros])
        vals=self.array('crossfit',[keys(d,'val'),*[xs[(a,None)] for a in ASPECTS],
                                    *[lab[a] for a in ASPECTS],[rt.name(r) for r in ros]],calc)
        rows={r:{a:float(vals[i,j]) for j,a in enumerate(ASPECTS)} for i,r in enumerate(ros)}
        pick,scores,guard=ns.select_pick(rows,ranges)
        table={rt.name(r):dict(map=rows[r],score=float(scores[r]),kept=bool(guard[r])) for r in ros}
        name=rt.name(pick)
        return dict(pick=name,table=table,unit=ranges,
                    unit_score=float(np.mean([rows[pick][a]/ranges[a] for a in ASPECTS])),n_val=len(keys(d,'val')))

    def pick(self,arm,d):
        if arm=='tpips':return self.pick_table(d,arm)
        sets=self.dependency(f'clay:{d}:val')['sets']
        tables={ts:self.pick_table(d,'clay',ts) for ts in sets}
        scores={ts:row['unit_score'] for ts,row in tables.items()}
        ts=choose_textset(scores,'ret')
        return dict(**tables[ts],text_set=ts,sets=tables,textset_scores=scores)

    def textset_clu(self,d):
        data=self.dependency(f'clay:{d}:val');y=labels(d,'val');scores={};per_aspect={}
        for ts in data['sets']:
            vals={}
            for a in ASPECTS:
                x=self.clay_vector(d,'val',ts,a)
                p=self.array('clay_val_parts',[keys(d,'val'),x,20,8],lambda:ns.partitions(x,20,d))
                vals[a]=float(ns.arm_metrics(p,y[a])[0])
            per_aspect[ts]=vals;scores[ts]=float(np.mean(list(vals.values())))
        return dict(set=choose_textset(scores,'clu'),scores=scores,per_aspect=per_aspect)

    def partitions_section(self,arm,d,e=None):
        self.dependency('counts');split=HELD[d];out={}
        ts=self.dependency(f'textset:clu:{d}')['set'] if arm=='clay' else None
        for a in ASPECTS:
            for k in sorted({K[d][a],20}):
                if arm=='cstar':
                    x=[self.vector(d,split,e,'cstar',a,s) for s in range(5)]
                    fn=lambda:np.stack([ns.partitions(v,k,d) for v in x])
                    inputs=[keys(d,split),*x,k,8]
                else:
                    x=self.vector(d,split,'qwen','tpips',a) if arm=='tpips' else self.clay_vector(d,split,ts,a)
                    fn=lambda:ns.partitions(x,k,d)
                    inputs=[keys(d,split),x,k,8]
                _,out[f'{a}/{k}']=self.stored('partitions',inputs,fn)
        return dict(arm=arm,set=d,encoder=e,text_set=ts,arrays=out)

    def get_pick(self,d,e,arm):
        return ns.reused_pick(d,e,'cstar') if arm=='cstar' else self.dependency(f'pick:{arm}:{d}')['pick']

    def cells(self,sc,d,e):
        ranges=self.dependency('ranges');out=[];families={}
        for c in cell_specs():
            if (c['sc'],c['set'],c['encoder'])!=(sc,d,e):continue
            a,k=c['aspect'],c['k'];y=labels(d,HELD[d])[a]
            if c['task']=='ret':
                cand,cp=self.ap(d,e,'cstar',a,y)
                comp,bp=self.ap(d,'clip_l14' if sc=='SC4' else e,c['comparator'],a,y)
                stats=ns.paired_retrieval(cand,comp,self.ret,apart=False)
                extra=dict(candidate_readout=cp,comparator_readout=bp,n_queries=cand.shape[-1])
            else:
                if (a,k) not in families:
                    if sc=='SC5':
                        cand=repeat_turtle(turtle_parts(d,e,'cstar',k,keys(d,HELD[d]),a))
                    else:
                        dep=self.dependency(f'parts:cstar:{d}:{e}')
                        cand=self.artifact(dep['arrays'][f'{a}/{k}'])
                    if sc in ('SC1','SC3'):
                        dep=self.dependency(f'parts:{c["comparator"]}:{d}')
                        comp=self.artifact(dep['arrays'][f'{a}/{k}'])
                    else:comp=turtle_parts(d,e,'plain',k,keys(d,HELD[d]))
                    families[a,k]=ns.joint_clustering(cand,comp,y,self.clu,self.workers)
                stats=families[a,k][c['metric']];extra=dict(n_labelled=int((y>=0).sum()))
            r=ranges[range_key(c)]['r']
            out.append({**c,**stats,**extra,'r':r,'r_source':range_source(c),'label':label(stats['lo'],r),'n_rows':len(y)})
        return out


def derive(sections,out=DEFAULT_OUT,clu=2000,ret=10000,workers=8):
    if not 1<=workers<=8 or not 1<=clu<=2000 or not 1<=ret<=10000:
        raise ValueError('workers 1..8, clustering draws 1..2000, retrieval draws 1..10000')
    if not sections or len(set(sections))!=len(sections) or set(sections)-set(SECTIONS):
        raise ValueError('unknown, duplicate or empty sections; run list')
    for section in sections:
        w=Work(out,clu,ret,workers);start=time.monotonic();b=section.split(':')
        for dependency in dependencies(section):w.dependency(dependency)
        print(f'C13 derive {section}',flush=True)
        if section in ('counts','ranges'):result=getattr(w,section)()
        elif b[0]=='clay':result=w.clay(*b[1:])
        elif b[0]=='pick':result=w.pick(*b[1:])
        elif b[0]=='textset':result=w.textset_clu(b[2])
        elif b[0]=='parts':result=w.partitions_section(*b[1:])
        else:result=w.cells(*b[1:])
        doc=dict(section=section,complete=True,source_sha256=source_identity(),draws_clu=clu,draws_ret=ret,
                 seconds=time.monotonic()-start,inputs=w.inputs,result=result)
        write_json(Path(out)/'derived'/f'{section}.json',doc)
        print(f'C13 done {section} seconds={doc["seconds"]:.1f}',flush=True)

# RESULT BOUNDARY: nothing above reads current-round picks, ranges, cells or status.


def derived_docs(out):
    docs={}
    for p in sorted((Path(out)/'derived').glob('*.json')):
        doc=read_json(p);s=doc.get('section')
        if s not in SECTIONS or s in docs or not doc.get('complete') or doc.get('source_sha256')!=source_identity():
            raise ValueError(f'unknown/duplicate/stale/incomplete section: {p}')
        docs[s]=doc
    return docs


def flat_picks(docs):
    picks={}
    for d in SETS:
        if f'pick:tpips:{d}' in docs:picks[f'tpips/{d}']=docs[f'pick:tpips:{d}']['result']['pick']
        if f'pick:clay:{d}' in docs:
            row=docs[f'pick:clay:{d}']['result']
            picks.update({f'clay/{d}':row['pick'],f'clay/{d}/text_set':row['text_set'],
                          f'clay_textset/ret/{d}':row['text_set']})
        if f'textset:clu:{d}' in docs:
            picks[f'clay_textset/clu/{d}']=docs[f'textset:clu:{d}']['result']['set']
    return picks


def finish(out=DEFAULT_OUT,records=ROUND/'records'):
    out=output_path(out);docs=derived_docs(out)
    status_path=Path(records)/'status.json';status=read_json(status_path)
    link=status['r17']
    path=Path(link['path'] if isinstance(link,dict) else link)
    previous=read_json(path)
    if isinstance(link,dict) and link.get('sha256')!=sha(path):raise ValueError('r17 status sha256 mismatch')
    rows=[decide(c,status,previous) for s,doc in docs.items() if s.startswith('cells:') for c in doc['result']]
    result=dict(picks=flat_picks(docs),cells=rows,source_sha256=source_identity(),
                status_sha256=sha(status_path),r17_status_sha256=sha(path),
                status_path=str(status_path.resolve()),r17_status_path=str(path.resolve()),
                derived_sha256={s:sha(out/'derived'/f'{s}.json') for s in docs})
    write_json(out/'c13_cells.json',result)
    print(f'C13 finish cells={len(rows)} picks={len(result["picks"])} void={sum(bool(c["void"]) for c in rows)}',flush=True)
    return result



class Comparison:
    def __init__(self):
        self.items=[]

    def add(self,name,derived,recorded,numeric=False):
        diff=None
        if numeric and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in (derived,recorded)):
            distance=abs(float(derived)-float(recorded))
            passed=bool(np.isfinite(distance) and distance<=TOL)
            diff=distance if np.isfinite(distance) else None
        else:
            passed=type(derived) is type(recorded) and derived==recorded
        self.items.append(dict(name=name,derived=derived,recorded=recorded,abs_diff=diff,**{'pass':passed}))

    def tree(self,name,derived,recorded):
        if isinstance(derived,dict):
            self.add(name+'/keys',sorted(derived),sorted(recorded) if isinstance(recorded,dict) else None)
            for key,value in derived.items():
                self.tree(name+'/'+key,value,recorded.get(key) if isinstance(recorded,dict) else None)
        elif isinstance(derived,list):
            self.add(name+'/length',len(derived),len(recorded) if isinstance(recorded,list) else None)
            for i,value in enumerate(derived):
                self.tree(name+'/'+str(i),value,recorded[i] if isinstance(recorded,list) and i<len(recorded) else None)
        else:self.add(name,derived,recorded,numeric=isinstance(derived,(float,int)) and not isinstance(derived,bool))

    def report(self):
        parts={}
        for item in self.items:
            p=parts.setdefault(item['name'].split('/')[0],dict(items=0,failed=0,max_abs=0.))
            p['items']+=1;p['failed']+=int(not item['pass'])
            p['max_abs']=max(p['max_abs'],item['abs_diff'] or 0.)
        return dict(items=self.items,items_compared=len(self.items),failed=sum(not x['pass'] for x in self.items),
                    max_abs=max((x['abs_diff'] or 0. for x in self.items),default=0.),parts=parts)


def full_scope(docs):
    if set(docs)!=set(SECTIONS):return False
    if any(not x.get('complete') or x.get('source_sha256')!=source_identity() for x in docs.values()):return False
    counts=docs['counts']['result']
    if counts!={d:{s:K[d] for s in ('val',HELD[d])} for d in SETS}:return False
    ranges=docs['ranges']['result']
    if set(ranges)!={range_key(c) for c in cell_specs()}:return False
    computed=[r for r in ranges.values() if r.get('source')=='computed/qwen_ami20']
    if len(computed)!=3 or any(len(r.get('per_seed',[]))!=5 for r in computed):return False
    cells=[c for s,doc in docs.items() if s.startswith('cells:') for c in doc['result']]
    if len(cells)!=337 or {c['id'] for c in cells}!={c['id'] for c in cell_specs()}:return False
    if any(c.get('n_draws')!=(2000 if c['task']=='clu' else 10000) for c in cells):return False
    if len(flat_picks(docs))!=10:return False
    for sc,d,e in UNITS:
        doc=docs[f'cells:{sc}:{d}:{e}']
        wanted={c['id'] for c in cell_specs() if (c['sc'],c['set'],c['encoder'])==(sc,d,e)}
        if {c['id'] for c in doc['result']}!=wanted:return False
        if doc.get('draws_clu' if SC[sc][0]=='clu' else 'draws_ret')!=(2000 if SC[sc][0]=='clu' else 10000):return False
    for d in SETS:
        for arm in ('tpips','clay'):
            row=docs[f'pick:{arm}:{d}']['result']
            tables={'tpips':row} if arm=='tpips' else row['sets']
            for ts,t in tables.items():
                width=4096 if arm=='tpips' else min(docs[f'clay:{d}:val']['result']['sets'][ts]['ranks'].values())
                if set(t['table'])!={rt.name(r) for r in rt.menu(width)}:return False
        for split in ('val',HELD[d]):
            data=docs[f'clay:{d}:{split}']['result']
            if 'A' not in data['sets'] or data['b_available']!=('B' in data['sets']):return False
            for row in data['sets'].values():
                if set(row['arrays'])!=set(ASPECTS):return False
        needed={f'{a}/{k}' for a in ASPECTS for k in {K[d][a],20}}
        for e in ENCODERS:
            row=docs[f'parts:cstar:{d}:{e}']['result']['arrays']
            if set(row)!=needed:return False
        for arm in ('tpips','clay'):
            row=docs[f'parts:{arm}:{d}']['result']['arrays']
            if set(row)!=needed:return False
    return True


def layout(node):
    if isinstance(node,dict):return {k:layout(v) for k,v in node.items()}
    if isinstance(node,list):return {'length':len(node),'item':layout(node[0]) if node else None}
    return type(node).__name__


def cell_rows(node):
    if isinstance(node,dict):
        if 'id' in node and 'sc' in node:yield node
        else:
            for value in node.values():yield from cell_rows(value)
    elif isinstance(node,list):
        for value in node:yield from cell_rows(value)


def compare_status_hashes(result, finished, records):
    """Bind labels to the exact status files read by finish, including old snapshots."""
    status_path=Path(finished.get('status_path',Path(records)/'status.json'))
    current=sha(status_path) if status_path.is_file() else None
    result.add('integrity/status_sha256',finished.get('status_sha256'),current)
    previous_path=finished.get('r17_status_path')
    if previous_path is None and status_path.is_file():
        link=read_json(status_path).get('r17')
        previous_path=link.get('path') if isinstance(link,dict) else link
    path=Path(previous_path) if isinstance(previous_path,str) else None
    current=sha(path) if path is not None and path.is_file() else None
    result.add('integrity/r17_status_sha256',finished.get('r17_status_sha256'),current)


def compare_turtle_inputs(result, root, out, full):
    """One exact input-hash check per sidecar; share hashes across K and starts."""
    paths=sorted((Path(root)/'parts/turtle').glob('*/*/*/*.json'))
    if full:result.add('turtle_inputs/sidecar_count',len(paths),412)
    work=Work(out);cache={}
    for path in paths:
        name='turtle_inputs/'+path.relative_to(Path(root)/'parts/turtle').as_posix()
        side=read_json(path)
        d,e,arm=path.relative_to(Path(root)/'parts/turtle').parts[:3]
        a,s=(side.get('aspect'),side.get('seed')) if arm=='cstar' else (None,None)
        if d not in SETS or e not in ENCODERS or arm not in ('plain','cstar') or (
                arm=='cstar' and (a not in ASPECTS or type(s) is not int or s not in range(5))):
            result.add(name,'valid TURTLE fit identity','invalid fit identity')
            continue
        tag=(d,e,arm,a,s)
        if tag not in cache:
            x=work.vector(d,HELD[d],e,'plain') if arm=='plain' else work.vector(
                d,HELD[d],e,'cstar',a,s,retrieval=True)
            h=hashlib.sha256(f'{x.dtype.str}{x.shape}'.encode())
            h.update(np.ascontiguousarray(x).tobytes())
            cache[tag]=h.hexdigest()
        result.add(name,cache[tag],side.get('input_sha256'))


def compare(root=DEFAULT_ROOT,records=ROUND/'records',out=DEFAULT_OUT,structure_only=False):
    """The sole current-round numeric-result reader; compare every supplied cell.

    A subset may omit derived units/picks, but cannot silently skip a cell that
    the requested reference root contains. Source/finish snapshots must agree.
    Paths and explanatory source/reason strings are not numeric verdict inputs.
    """
    root,records,out=Path(root),Path(records),output_path(out)
    if not (out/'c13_cells.json').is_file():raise ValueError('run finish before compare')
    paths=[root/'stage2/cells.json',records/'stage1_om_picks.json',records/'stage1_om_ranges.json']
    paths+=sorted((root/'stage1/tables').glob('*.json'))
    if structure_only:
        for path in paths:
            if path.is_file():print(str(path)+': '+json.dumps(layout(read_json(path)),sort_keys=True))
            else:print(str(path)+': absent')
        return {'failed':0}
    docs=derived_docs(out);finished=read_json(out/'c13_cells.json')
    references=list(cell_rows(read_json(root/'stage2/cells.json')))
    refs={c['id']:c for c in references};ours={c['id']:c for c in finished['cells']}
    recorded_picks=read_json(records/'stage1_om_picks.json')
    recorded_ranges=read_json(records/'stage1_om_ranges.json')
    result=Comparison();skipped=[]
    expected={c['id'] for c in cell_specs()}
    full=full_scope(docs) and set(refs)==expected and all(
        c.get('n_draws')==(2000 if c.get('task')=='clu' else 10000) for c in refs.values())
    scope='full' if full else 'subset'
    result.add('integrity/source_sha256',finished.get('source_sha256'),source_identity())
    compare_status_hashes(result,finished,records)
    compare_turtle_inputs(result,root,out,full)
    result.tree('integrity/derived_snapshot',finished.get('derived_sha256'),
                {s:sha(out/'derived'/f'{s}.json') for s in docs})
    result.tree('integrity/pick_snapshot',finished.get('picks'),flat_picks(docs))
    derived_cells={c['id']:c for s,doc in docs.items() if s.startswith('cells:') for c in doc['result']}
    result.add('integrity/finish_cell_ids',sorted(ours),sorted(derived_cells))
    for cid,c in ours.items():
        source=derived_cells.get(cid,{})
        for field in ('cand','comp','d','lo','hi','r','n_draws'):
            result.add('integrity/'+cid+'/'+field,c.get(field),source.get(field),numeric=field!='n_draws')
    result.add('cells/unique_reference_ids',len(refs),len(references))
    result.add('cells/unique_derived_ids',len(ours),len(finished['cells']))
    result.add('cells/reference_nonempty',bool(refs),True)
    result.add('cells/unknown_reference_ids',sorted(set(refs)-expected),[])
    result.add('cells/covered_reference_ids',sorted(set(ours)&set(refs)),sorted(refs))
    for cid,ref in refs.items():
        if cid not in ours:continue
        c=ours[cid];prefix='cells:'+c['sc']+':'+c['set']+':'+c['encoder']+'/'+cid
        for field in ('cand','comp','d','lo','hi','r'):
            value=c.get('r_used') if field=='r' else c.get(field)
            result.add(prefix+'/'+field,value,ref.get(field),numeric=True)
        for field in ('id','sc','set','encoder','aspect','task','metric','k','candidate','comparator','n_draws','n_rows','label'):
            result.add(prefix+'/'+field,c.get(field),ref.get(field))
        result.add(prefix+'/void',bool(c.get('void')),bool(ref.get('void')))
        if c['task']=='ret':
            for field in ('candidate_readout','comparator_readout'):
                result.add(prefix+'/'+field,c.get(field),ref.get(field))
    if 'counts' in docs:
        result.tree('counts/binding_K',docs['counts']['result'],{d:{split:K[d] for split in ('val',HELD[d])} for d in SETS})
    if 'ranges' in docs:
        for key,row in docs['ranges']['result'].items():
            ref=recorded_ranges.get(key,{})
            result.add('ranges/'+key+'/r',row['r'],ref.get('r'),True)
            if row['source']=='computed/qwen_ami20':
                theirs=ref.get('per_seed')
                if isinstance(theirs,dict):theirs=[theirs.get(str(i)) for i in range(5)]
                result.tree('ranges/'+key+'/per_seed',row['per_seed'],theirs)
    for key,value in finished['picks'].items():
        if key.endswith('/text_set'):
            ref=recorded_picks.get(key[:-9],{}).get('text_set')
        else:
            row=recorded_picks.get(key,{})
            ref=row.get('set') if key.startswith('clay_textset/') else row.get('pick')
        result.add('picks/'+key,value,ref)
    compared_tables=[]
    def compare_table(d,arm,row,ts=None):
        token=f'{arm}__{d}'+(f'__{ts}' if ts else '')
        path=root/'stage1/tables'/(token+'.json')
        table=read_json(path) if path.is_file() else {}
        prefix='tables:'+arm+':'+d+((':'+ts) if ts else '')
        result.add(prefix+'/present',bool(table),True)
        result.tree(prefix+'/table',row['table'],table.get('table'))
        result.add(prefix+'/pick',row['pick'],table.get('pick'))
        result.tree(prefix+'/ranges',row['unit'],table.get('ranges'))
        result.add(prefix+'/n_val',row['n_val'],table.get('n_val'))
        compared_tables.append(token+'.json')
    for d in SETS:
        for arm in ('tpips','clay'):
            section=f'pick:{arm}:{d}'
            if section not in docs:continue
            row=docs[section]['result'];ref=recorded_picks.get(f'{arm}/{d}',{})
            if arm=='tpips':compare_table(d,arm,row)
            else:
                for ts,t in row['sets'].items():compare_table(d,arm,t,ts)
                result.add('textset:ret:'+d+'/set',row['text_set'],recorded_picks.get(f'clay_textset/ret/{d}',{}).get('set'))
                score_ref=recorded_picks.get(f'clay_textset/ret/{d}',{}).get('scores',{})
                result.add('textset:ret:'+d+'/sets',sorted(row['sets']),sorted(score_ref))
                data=docs[f'clay:{d}:val']['result']
                for ts,t in row['sets'].items():
                    r=score_ref.get(ts,{})
                    prefix='textset:ret:'+d+'/'+ts
                    result.add(prefix+'/pick',t['pick'],r.get('pick'))
                    result.add(prefix+'/score',t['unit_score'],r.get('score'),True)
                    result.tree(prefix+'/map',t['table'][t['pick']]['map'],r.get('map'))
                    result.tree(prefix+'/r0_map',t['table']['none+k0']['map'],r.get('r0_map'))
                    result.add(prefix+'/rank',min(data['sets'][ts]['ranks'].values()),r.get('rank'))
                result.add('picks/clay/'+d+'/unit_score',row['unit_score'],ref.get('unit_score'),True)
            prefix='pick-record:'+arm+':'+d
            for field in ('pick','n_val'):result.add(prefix+'/'+field,row[field],ref.get(field))
            w,k=row['pick'].split('+k')
            result.add(prefix+'/w',w,ref.get('w'));result.add(prefix+'/k',int(k),ref.get('k'))
            result.tree(prefix+'/unit',row['unit'],ref.get('unit'))
            want={**row['table'][row['pick']],'r0_map':row['table']['none+k0']['map']}
            result.tree(prefix+'/scores',want,ref.get('scores'))
        section=f'textset:clu:{d}'
        if section in docs:
            row=docs[section]['result'];ref=recorded_picks.get(f'clay_textset/clu/{d}',{})
            prefix='textset:clu:'+d
            result.add(prefix+'/sets',sorted(row['scores']),sorted(ref.get('scores',{})))
            for ts,score in row['scores'].items():
                result.add(prefix+'/'+ts+'/mean_ami20',score,ref.get('scores',{}).get(ts,{}).get('mean_ami20'),True)
                result.tree(prefix+'/'+ts+'/ami20',row['per_aspect'][ts],ref.get('scores',{}).get(ts,{}).get('ami20'))
    if not full:
        skipped=['sections not derived: '+', '.join(s for s in SECTIONS if s not in docs),
                 f'cells outside reference subset: {len(expected-set(refs))}',
                 'pick keys not derived: '+', '.join(sorted(({f'{kind}/{d}' for d in SETS for kind in
                      ('tpips','clay','clay_textset/clu','clay_textset/ret')}|{f'clay/{d}/text_set' for d in SETS})-set(finished['picks']))),
                 'tables not compared: '+', '.join(p.name for p in sorted((root/'stage1/tables').glob('*.json')) if p.name not in compared_tables)]
        if any(c['n_draws']!=(2000 if c['task']=='clu' else 10000) for c in ours.values()):
            skipped.append('full bootstrap counts (2000 clustering / 10000 retrieval); subset draws only')
    report={**result.report(),'scope':scope,'subset':sorted(refs) if not full else None,'skipped':skipped,
            'draw_counts':{s:{'clu':doc['draws_clu'],'ret':doc['draws_ret']} for s,doc in docs.items()},
            'reference_draw_counts':{'clu':sorted({c.get('n_draws') for c in refs.values() if c['task']=='clu'}),
                                     'ret':sorted({c.get('n_draws') for c in refs.values() if c['task']=='ret'})}}
    write_json(out/'compare.json',report)
    print(f'C13 {"FAIL" if report["failed"] else "PASS"} items={report["items_compared"]} failed={report["failed"]} max_abs={report["max_abs"]:.12g} scope={scope}',flush=True)
    return report


def estimate(section):
    """Conservative full-draw slot-minutes, calibrated from the six-unit smoke."""
    if section=='counts':return 1
    if section=='ranges':return 2
    if section.startswith('clay:'):return 1
    if section.startswith('pick:tpips:'):return 5
    if section.startswith('pick:clay:'):return 3
    if section.startswith('textset:'):return 1
    if section.startswith('parts:cstar:'):return 8
    if section.startswith('parts:tpips:'):return 3
    if section.startswith('parts:clay:'):return 1
    _,sc,d,_=section.split(':')
    if sc=='SC2':return 5 if d=='artelingo' else 8
    if sc=='SC4':return 4 if d=='artelingo' else 7
    return 9


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('list')
    d=sub.add_parser('derive');d.add_argument('--sections',nargs='+',required=True)
    d.add_argument('--out',type=Path,default=DEFAULT_OUT)
    d.add_argument('--draws-clu',type=int,default=2000);d.add_argument('--draws-ret',type=int,default=10000)
    d.add_argument('--workers',type=int,default=8)
    for name in ('finish','compare'):
        s=sub.add_parser(name);s.add_argument('--out',type=Path,default=DEFAULT_OUT)
        s.add_argument('--records',type=Path,default=ROUND/'records')
        if name=='compare':
            s.add_argument('--root',type=Path,default=DEFAULT_ROOT);s.add_argument('--structure-only',action='store_true')
    args=p.parse_args(argv)
    try:
        with threadpool_limits(8):
            if args.cmd=='list':
                for s in SECTIONS:print(f'{s} --est {estimate(s)}m')
            elif args.cmd=='derive':derive(args.sections,args.out,args.draws_clu,args.draws_ret,args.workers)
            elif args.cmd=='finish':finish(args.out,args.records)
            else:return int(bool(compare(args.root,args.records,args.out,args.structure_only)['failed']))
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print(f'C13 ERROR: {exc}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
