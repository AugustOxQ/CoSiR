"""Independent r19 C13, from §3.2--3.5 and §6 and saved, keyed inputs.

No r19 module or scorer is imported. bs_c13/mt_c13 supply only frozen
numerical array functions. ArtELingo and Affection retain their inherited
Python-sorted painting/key order (rt_data/ac_data's explicit interfaces);
MMArt and SemArt use SHA256 key order. Frozen retrieval is ac_data._unit(raw)
on ArtELingo and Affection (bs_c13.py:373, 388, 411, 430--431, 530), and
unit(unit(raw)) on MMArt and SemArt. ArtELingo/Affection teachers reuse saved
per-query AP (bs_c13.py:391, 433; r11_ap/saved_ap), not recomputed retrieval.
c* retrieval is unit(export); clustering is raw c* / once-normalised stores.

`sections` prints the dependency-first full schedule. Each section is atomic,
resumable and independently queueable. `smoke:*` are real-population subsets,
with the real draws; never a certification. `merge` is deliberately full-only;
`compare` assembles available sections for a subset without requiring merge.
All caches bind array bytes, keys, labels, threads and frozen source bytes.
Never run a helper CLI/derive/compare or a helper with a default write path.
"""
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROUND = Path('/project/CoSiR/.claude/worktrees/r19-int/src/test/20261103_past_picks')
MAIN = Path('/project/CoSiR/src/test')
BASE = Path('/data/SSD2/cosir_r19/20261103_past_picks')
DEFAULT_ROOT, DEFAULT_OUT, SMOKE_OUT = BASE/'outputs', BASE/'c13', BASE/'c13_smoke'
RECORDS = ROUND/'records'
PRE = Path('/data/SSD2/pre_extract')
R10 = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full')
R11 = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R12 = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R14 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R15 = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
INDEX = Path('/data/SSD/mmart/mmart_index.parquet')
NEAR = Path('/data/SSD/mmart/mmart_nearcopy.parquet')
SEMART = Path('/data/SSD/semart/SemArt')
SC = MAIN/'20261103_semart_check'
ENCODERS = ('qwen','siglip2','siglip1','clip_l14','dinov2_large','clip_b32','clip_b16')
NEW = ENCODERS[4:]
ASPECTS = ('emotion','style','genre')
SEEDS = tuple(range(5))
FOLDERS = dict(zip(ENCODERS, ('qwen3vl_emb_2b','siglip2_so400m','siglip_so400m','clip_l14','dinov2_large','clip_b32','clip_b16')))
MATCHED = dict(zip(ENCODERS, ('P3','M2','M8','M7','M5','M3','M6')))
TEACHER = dict(zip(ASPECTS, ('emotion_focus','style_word','genre_word')))
TYPES = ('genre','historical','interior','landscape','mythological','other','portrait','religious','still-life','study')
for folder in ('20261102_backbones_sweep','20261102_mmart_test','20261102_mmart_prep'):
    sys.path.insert(0, str(MAIN/folder))
import bs_c13 as frozen
import mt_c13 as mm
ac_cluster, ac_select, ac_stats = frozen.ac_cluster, frozen.ac_select, frozen.ac_stats
rt_data, ac_data = frozen.rt_data, frozen.ac_data
rt_metrics, rt_readout, rt_stats = frozen.rt_metrics, frozen.rt_readout, frozen.rt_stats
join_rows, hashed_order = frozen.join_rows, frozen.hashed_order
unit, store_retrieval = mm.unit, mm.store_retrieval
cell_outcome = mm.cell_outcome
TOL = 1e-6
IDENTITY = ('sc','encoder','aspect','task','data','candidate','comparator')
DATASETS = dict(art='artelingo_held',aff='affection_test',mm='mmart_test',sem='semart_new',cost='cost')


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def keys_sha(keys):
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def check_output(path):
    p = Path(path).resolve()
    if not any(p.is_relative_to(x) for x in (DEFAULT_OUT,SMOKE_OUT)):
        raise ValueError(f'output outside C13 roots: {p}')
    return p


def atomic_json(path, doc):
    path = check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False, default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    os.replace(tmp,path)


@lru_cache(None)
def source_identity():
    # Compare/layout-only fixes do not invalidate expensive numerical caches.
    # Bind exact source segments of every derivation function and constant,
    # plus the complete helper/frozen sources; retain whole-file provenance
    # separately in section files.
    source=Path(__file__).read_text();tree=ast.parse(source)
    interface={'Comparison','compare','check_outcomes','summary','result_paths','record_refusal','main'}
    segments=[ast.get_source_segment(source,n) for n in tree.body
              if not isinstance(n,(ast.FunctionDef,ast.ClassDef)) or n.name not in interface]
    return hashlib.sha256(('\n'.join(segments)+file_sha(mm.__file__)+frozen.frozen_identity()).encode()).hexdigest()


# SemArt's name rule is transcribed from sc_check (no import of its scorers).
SUFFIXES = ('the younger','the elder','il giovane','il vecchio','jr','sr','ii','iii')
ALIASES = {'RAFFAELLO Sanzio':'raphael','TIZIANO Vecellio':'titian'}


def slug(text):
    text = unicodedata.normalize('NFKD',str(text)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9?]+','-',text).strip('-')


def variants(author):
    a = str(author).strip()
    if ',' not in a:
        s = slug(a); toks = s.split('-')
        return {s,toks[0]},toks[:1],None
    last,first = (p.strip() for p in a.split(',',1))
    given,suffix = first,''
    for suf in SUFFIXES:
        if first.lower().endswith(' '+suf) or first.lower()==suf:
            given,suffix = first[:-len(suf)].strip(),first[-len(suf):]
            break
    out = {slug(f'{first} {last}'),slug(f'{given} {last} {suffix}'),slug(last)}
    return out-{''},slug(last).split('-'),slug(given).split('-')[0] if slug(given) else None


def wild(token,tokens):
    if '?' not in token:
        return token in tokens
    rx = re.compile('^'+'.{0,2}'.join(re.escape(p) for p in token.split('?'))+'$')
    return any(rx.match(t) for t in tokens)


def artist_tier(author,artists):
    if ALIASES.get(author) in artists:
        return 'exact'
    candidates,surname,gtok = variants(author)
    if candidates & artists:
        return 'exact'
    sur = [t for t in surname if t][:1]
    hits = [a for a in artists if all(wild(t,set(a.split('-'))) for t in sur)] if sur else []
    if hits and (gtok is None or any(wild(gtok,set(a.split('-'))) for a in hits)):
        return 'given_surname'
    return 'surname_only' if hits else 'none'


def semart_rule(table,artists,md5_of):
    """Toy-testable gate, then md5 first in SHA256 order. Inputs join by key."""
    assert table.image_file.is_unique
    tiers = {a:artist_tier(a,artists) for a in set(table.author)}
    tier = table.author.map(tiers)
    match = bool((tier == table.artist_tier).all()) if 'artist_tier' in table else True
    assert match, 'recomputed artist tiers differ from saved input'
    mask = (table.max_cos < .95) & (tier=='none') & ~table.author.str.startswith('UNKNOWN')
    rows = table.set_index('image_file')
    gate = hashed_order(table.loc[mask,'image_file'])
    groups = defaultdict(list)
    for key in gate:
        groups[md5_of(key)].append(key)
    kept = [k for k in gate if groups[md5_of(k)][0]==k]
    duplicates = [dict(md5=m,keep=ks[0],dropped=ks[1:]) for m,ks in groups.items() if len(ks)>1]
    close_gate = [k for k in gate if rows.loc[k,'max_cos']>=.93]
    close = [k for k in kept if rows.loc[k,'max_cos']>=.93]
    far = [k for k in kept if rows.loc[k,'max_cos']<.93]
    types = lambda ks:dict(Counter(rows.loc[ks,'type']))
    artists_n = lambda ks:len(set(rows.loc[ks,'author']))
    g = dict(paintings=len(gate),artists=artists_n(gate),types=types(gate),tiers_match_parquet=match,n_set=len(kept),md5_groups=duplicates)
    return dict(gate=g,n_gate=len(gate),n_set=len(kept),artists_set=artists_n(kept),md5_groups=duplicates,
                n_md5_dropped=len(gate)-len(kept),close_before_md5=len(close_gate),close=len(close),far=len(far),
                close_cut=.93,set_types=types(kept),keys_sha256={n:keys_sha(ks) for n,ks in [('gate',gate),('set',kept),('close',close),('far',far)]},
                keys=kept,gate_keys=gate)


# The schedule deliberately keeps all dependencies before scoring sections.
def schedule():
    result = [('set',1)]
    for e in NEW:
        result += [(f'ranges:{e}:{d}',3 if d!='aff' else 2) for d in ('art','aff','mm')]
        result += [(f'picks:{e}:{d}',12 if d!='aff' else 3) for d in ('art','aff','mm')]
    result += [(f'semval:{e}',3 if e in ENCODERS[:4] else 0.2) for e in ENCODERS]
    for e in NEW:
        result += [(f'cells:{e}:{d}:{t}',12 if t=='clu' else 3) for d in ('art','mm') for t in ('clu','ret')]
        result += [(f'cells:{e}:aff',4)]
    result += [(f'cells:{e}:sem:{t}',4 if t=='clu' else 2) for e in ENCODERS for t in ('clu','ret')]
    return result+[('cost',.1)]


FULL_SECTIONS = tuple(n for n,_ in schedule())
SMOKE_SECTIONS = ('set','ranges:clip_b32:art','ranges:clip_b32:aff','ranges:clip_b32:mm',
                  'picks:clip_b32:art','picks:clip_b32:aff','picks:clip_b32:mm','semval:clip_b32',
                  'smoke:clip_b32:art:clu','smoke:clip_b32:art:ret','smoke:clip_b32:aff',
                  'smoke:clip_b32:mm:clu','smoke:clip_b32:mm:ret','cells:clip_b32:sem:clu','cells:clip_b32:sem:ret','cost')


class Work:
    def __init__(self,root=DEFAULT_ROOT,out=DEFAULT_OUT,n_clu=2000,n_ret=10000,threads=8):
        assert isinstance(threads,int) and not isinstance(threads,bool) and threads>0
        self.root,self.out = Path(root).resolve(),check_output(out)
        self.n_clu,self.n_ret,self.threads = n_clu,n_ret,threads
        self.inputs = {}
        self._sets = {}

    def track(self,path):
        path=Path(path)
        self.inputs[str(path)] = file_sha(path)
        return path

    def json(self,path):
        return read_json(self.track(path))

    def array(self,name,inputs,fn):
        h=hashlib.sha256(source_identity().encode())
        for x in inputs:
            if isinstance(x,np.ndarray):
                h.update(str((x.shape,x.dtype.str)).encode());h.update(np.ascontiguousarray(x).tobytes())
            else:
                h.update(json.dumps(x,sort_keys=True).encode())
        path=self.out/'cache'/f'{name}__{h.hexdigest()}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:return z['value']
        start=time.monotonic(); value=np.asarray(fn())
        assert np.isfinite(value).all(),name
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(tmp,value=value);os.replace(tmp,path)
        print(f'{name} {time.monotonic()-start:.1f}s',flush=True)
        return value

    def parts(self,x,keys):
        return self.array('parts',[keys,x,self.threads],lambda:mm.kmeans_partitions(x,self.threads))

    def dep(self,name):
        p=self.out/'sections'/f'{name}.json'
        if not p.exists():raise ValueError(f'missing dependency {name}; derive it first')
        doc=read_json(p)
        # A dependency is an immutable, independently derived input. Its code
        # provenance remains in its own header; changing a consumer does not
        # require re-scoring an unchanged validation range. Bind its exact
        # section bytes and validate every saved input instead.
        self.validate(doc,require_current=False)
        self.track(p)
        for path,digest in doc['inputs'].items():
            if path in self.inputs:assert self.inputs[path]==digest,'conflicting input bytes'
            self.inputs[path]=digest
        return doc['result']

    def validate(self,doc,require_current=True):
        assert doc['complete'] and doc['root']==str(self.root),'incomplete or wrong root'
        assert (doc['draws_clu'],doc['draws_ret'],doc['kmeans_threads'])==(self.n_clu,self.n_ret,self.threads),'draw counts or threads differ'
        if require_current:
            assert doc['source_sha256']==source_identity(),'C13 source changed; rerun section with --force'
        for p,h in doc['inputs'].items():assert file_sha(p)==h,f'input bytes changed: {p}'

    def semset(self):
        table=pd.read_parquet(self.track(SC/'sc_nearcopy.parquet'))
        # Frozen r10 split gives all train painting identities, never a score.
        split=rt_data.ct2_data.split2('full')
        train=split['all_painting_ids'][split['fit']].tolist()
        assert len(train)==42969
        artists={n.split('_')[0] for n in train}
        csv=pd.concat([pd.read_csv(self.track(SEMART/f'semart_{s}.csv'),sep='\t',encoding='latin-1') for s in ('train','val','test')]).set_index('IMAGE_FILE')
        assert csv.index.is_unique
        aligned=csv.loc[table.image_file]
        assert aligned.AUTHOR.tolist()==table.author.tolist() and aligned.TYPE.tolist()==table.type.tolist()
        digests={}
        def md5(key):
            if key not in digests:
                path=self.track(SEMART/'Images'/key)
                digests[key]=hashlib.md5(path.read_bytes()).hexdigest()
            return digests[key]
        result=semart_rule(table,artists,md5)
        expected=self.json(SC/'sc_check.json')
        result['gate'].update(expected=dict(paintings=11162,artists=2717,types=expected['new_artist_types']),n_set_expected=11159,
                              n_set_as_expected=result['n_set']==11159)
        g=result['gate'];g['pass']=(g['paintings']==11162 and g['artists']==2717 and g['types']==expected['new_artist_types'] and g['n_set_as_expected'])
        # Count internal high-cosine pairs, without removing them.
        x=unit(self.store(PRE/'semart/qwen3vl_emb_2b',result['keys'],'image_file'),floor=True)
        def pairs():
            count=0;members=set()
            for start in range(0,len(x),1024):
                sim=x[start:start+1024]@x.T
                for i in range(len(sim)):sim[i,:start+i+1]=-1
                ii,jj=np.nonzero(sim>=.95);count+=len(ii);members.update((ii+start).tolist());members.update(jj.tolist())
            return [count,len(members)]
        count,members=map(int,self.array('inner_pairs',[result['keys'],x],pairs))
        result['inner_pairs_ge_0.95']=dict(pairs=count,paintings=members,n=len(x))
        return result

    def population(self,d,split):
        k=(d,split)
        if k in self._sets:return self._sets[k]
        if d=='art':
            keys=rt_data.names('val' if split=='val' else 'held');labels=rt_data.labels('val' if split=='val' else 'held')
        elif d=='aff':
            keys=ac_data.keys(split);labels={'emotion':ac_data.labels(split)}
        elif d=='mm':
            table=pd.read_parquet(self.track(INDEX));assert table.key.is_unique
            if split=='val':keys=hashed_order(table.loc[table.split=='val','key'])
            else:keys=mm.cut(table,pd.read_parquet(self.track(NEAR)),.95)['keys']
            rows=table.set_index('key').loc[keys];labels={}
            for a in ASPECTS:
                values=rows['genre_name' if a=='genre' else a].fillna('');codes={v:i for i,v in enumerate(sorted(set(values)-{''}))}
                labels[a]=np.array([codes.get(v,-1) for v in values],np.int64)
        else:
            if split=='val':return self.population('art','val')
            keys=self.dep('set')['keys'];table=pd.read_parquet(self.track(SC/'sc_nearcopy.parquet')).set_index('image_file')
            labels={'genre':np.array([TYPES.index(t) for t in table.loc[keys,'type']],np.int64)}
        assert len(keys)==len(set(keys))
        self._sets[k]=(keys,labels)
        return keys,labels

    def croot(self,e):
        return R10 if e=='qwen' else (R14 if e in ENCODERS[:4] else self.root)/'cstar'/e

    def export(self,path,keys,array='emb',positional=None):
        path=self.track(path);digest=self.inputs[str(path)]
        checked=0
        with np.load(path,allow_pickle=False) as z:
            x=z[array]
            stored=z['keys'].tolist() if 'keys' in z else positional
        for p in (path.with_suffix('.json'),path.with_suffix('.pp.json')):
            if not p.exists():continue
            side=self.json(p)
            for field in ('sha256','output_sha256'):
                if field in side:
                    # r10/r14's legacy .json hashes the array; r19's .pp.json
                    # and the keyed-export .json hash the complete NPZ bytes.
                    legacy_array=('arrays' in side or positional is not None and array=='prob' and not p.name.endswith('.pp.json'))
                    expected=frozen.rs_vh.array_sha(x) if field=='sha256' and legacy_array else digest
                    assert side[field]==expected,f'export checksum: {p}'
                    checked+=1
            if array in side.get('arrays',{}):
                meta=side['arrays'][array]
                assert meta==dict(shape=list(x.shape),dtype=str(x.dtype),sha256=frozen.rs_vh.array_sha(x)),f'array checksum: {p}'
                checked+=1
        assert checked,'export sidecars contain no checked digest'
        assert stored is not None,'no asserted key list'
        # Positional-only r10 exports carry the list identity in the sidecar.
        side=self.json(path.with_suffix('.json'))
        if positional is not None:
            target=side.get('list_sha256',side.get('config',{}).get('names_sha256',{}).get('dev'))
            assert target is not None and target==frozen.rs_vh.names_sha(stored),'positional export lacks its key-list identity'
        if 'keys_sha256' in side:assert side['keys_sha256']==keys_sha(stored)
        assert len(stored)==len(x) and np.isfinite(x).all()
        return x[join_rows(stored,keys)].astype(np.float32)

    def store(self,folder,keys,field):
        folder=Path(folder)
        if (folder/'keys.json').exists():stored=self.json(folder/'keys.json')
        elif (folder/'ids.npz').exists():
            with np.load(self.track(folder/'ids.npz'),allow_pickle=False) as z:stored=z['painting_names'].tolist()
        else:
            images=self.json(folder/'index.json')['images']
            stored=[f"{r['source']}/{r['image_name']}" if field=='aff' else r[field] for r in images]
        path=self.track(folder/'img.npy')
        x=np.load(path,mmap_mode='r',allow_pickle=False)
        assert len(x)==len(stored) and x.dtype==np.float16
        return x[join_rows(stored,keys)].astype(np.float32)

    def vector(self,e,d,arm,split,a,seed=None,retrieval=False):
        keys,_=self.population(d,split)
        if arm in ('cstar','photo'):
            if d in ('art','sem') and split=='val':
                graph=dict(emotion='Gres',style='Fres05',genre='R0o')[a]
                p=self.croot(e)/f'train/k11=40/R6-5/{a}__{graph}__s{seed}.npz'
                x=self.export(p,keys,'img',rt_data.names('val'))
            elif d=='art':x=self.export(self.croot(e)/f'held/k11=40/{a}__s{seed}.npz',keys,'img',rt_data.names('held'))
            elif d=='aff':x=self.export(self.root/f'exports/affection/photo_{e}/{split}__s{seed}.npz',keys)
            elif d=='mm':x=self.export(self.root/f'mmart/exports/{"val" if split=="val" else "new"}/{e}/cstar/{a}__s{seed}.npz',keys)
            else:x=self.export(self.root/f'exports/semart/{e}/cstar/genre__s{seed}.npz',keys)
            return unit(x) if retrieval else x
        teacher=arm=='teacher';foldername=f'qwen3vl_emb_2b_aspect_prompts/{TEACHER[a]}' if teacher else FOLDERS[e]
        if d in ('art','sem') and (split=='val' or d=='art'):
            folder=PRE/'artelingo_full'/(foldername+('_s2img' if split=='val' else '_held') if teacher else foldername)
            x=self.store(folder,keys,'painting')
        elif d=='sem':x=self.store(PRE/'semart'/foldername,keys,'image_file')
        elif d=='aff':
            folder=PRE/'affection_local'/foldername
            x=self.store(folder,keys,'aff')
        else:
            table=pd.read_parquet(self.track(INDEX)).set_index('key').loc[keys]
            x=None
            for group in ('none','trainval','held'):
                positions=np.flatnonzero(table.artelingo.to_numpy()==group)
                if not len(positions):continue
                ns=[keys[i] for i in positions] if group=='none' else table.iloc[positions].painting.tolist()
                folder=PRE/('mmart' if group=='none' else 'artelingo_full')/(foldername if group=='none' or not teacher else foldername+('_s2img' if group=='trainval' else '_held'))
                values=self.store(folder,ns,'painting' if group!='none' else 'key')
                if x is None:x=np.empty((len(keys),values.shape[1]),np.float32)
                x[positions]=values
        if retrieval and arm=='frozen' and d in ('art','aff'):
            return ac_data._unit(x)
        return store_retrieval(x) if retrieval else unit(x,floor=True)

    def matched(self,e,d):
        keys,_=self.population(d,'test');out={}
        for s in SEEDS:
            if d=='art':
                base=self.root/'matched/part2' if e=='dinov2_large' else (MAIN/'20261025_recipe_backbones/outputs/part2' if e=='clip_b32' else Path('/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/part2'))
                p=base/f'full/held/{MATCHED[e]}/P3__s{s}.npz'
                prob=self.export(p,keys,'prob',rt_data.names('held'))
            else:prob=self.export(self.root/f'exports/{"semart" if d=="sem" else "mmart_new"}/{e}/matched/probs__s{s}.npz',keys,'probs')
            out[s]=dict(own=np.argmax(prob,axis=1).astype(np.int32)[None,:],r20=self.parts(unit(prob),keys))
        return out

    def raw_ap(self,x,y,keys):
        def calc():
            result=rt_metrics.retrieval(x,y,qe_k=0,workers=8)
            assert np.array_equal(result['rows'],np.flatnonzero(y>=0))
            return result['ap']
        return self.array('r0_ap',[keys,x,y],calc)

    def ranges(self,e,d):
        keys,ys=self.population(d,'val');result={}
        for a,y in ys.items():
            result[a]={}
            for task in ('clu','ret'):
                values=[]
                for s in SEEDS:
                    x=self.vector(e,d,'photo' if d=='aff' else 'cstar','val',a,s,retrieval=task=='ret')
                    values.append(float(ac_cluster.ami(self.parts(x,keys),y)) if task=='clu' else float(self.raw_ap(x,y,keys).mean()))
                result[a][task]=dict(per_seed={str(s):v for s,v in zip(SEEDS,values)},r=rt_stats.seed_range(values))
            if d=='aff':
                values=[]
                dist=ac_data.emotion_dist('val')
                for s in SEEDS:
                    x=self.vector(e,d,'photo','val',a,s,retrieval=True)
                    values.append(float(self.array('ndcg',[keys,x,dist],lambda:rt_metrics.graded(x,dist,workers=8)).mean()))
                result[a]['ndcg']=dict(per_seed={str(s):v for s,v in zip(SEEDS,values)},r=rt_stats.seed_range(values))
        return result

    def picks(self,e,d):
        ranges=self.dep(f'ranges:{e}:{d}');keys,ys=self.population(d,'val');result={}
        for arm in ('cstar','frozen'):
            rr={a:ranges[a]['ret']['r'] for a in ys}
            if arm=='frozen' and d=='art':
                old=self.json(MAIN/'20261029_retrieval/stage0a_ranges.json')['map'];rr={a:old[a]['r'] for a in ys}
            if arm=='frozen' and d=='aff':rr={'emotion':self.json(R12/'stage0b/ranges.json')['map']['r']}
            scores=[]
            for s in SEEDS if arm=='cstar' else (None,):
                aspects=[]
                for a,y in ys.items():
                    x=self.vector(e,d,'photo' if d=='aff' and arm=='cstar' else arm,'val',a,s,retrieval=True)
                    ros=rt_readout.menu(x.shape[1])
                    def calc():
                        rows=rt_readout.crossfit_all({(a,None):x},{a:y},ros,workers=8)
                        return [[rows[ro][a]] for ro in ros]
                    aspects.append(self.array('crossfit',[keys,x,y],calc)[:,0])
                scores.append(np.stack(aspects,axis=1))
            means=np.mean(scores,axis=0)
            rows={ro:{a:float(means[i,j]) for j,a in enumerate(ys)} for i,ro in enumerate(ros)}
            if d=='art':pick=rt_readout.select(rows,rr);sc=rt_readout.score(rows,rr);keep=rt_readout.guard(rows,rr)
            else:pick=ac_select.pick_readout(rows,rr);sc=ac_select.readout_scores(rows,rr);keep=ac_select.readout_guard(rows,rr)
            result[arm]=dict(pick=rt_readout.name(pick),table={rt_readout.name(ro):dict(map=rows[ro],score=float(sc[ro]),kept=bool(keep[ro])) for ro in ros},ranges=rr,dropped=rt_readout.half_dropped(ys))
        return result

    def semval(self,e):
        if e in NEW:
            ranges=self.dep(f'ranges:{e}:art')['genre'];p=self.dep(f'picks:{e}:art')
        else:
            keys,ys=self.population('art','val');y=ys['genre'];ranges={}
            for task in ('clu','ret'):
                values=[]
                for s in SEEDS:
                    x=self.vector(e,'art','cstar','val','genre',s,retrieval=task=='ret')
                    values.append(float(ac_cluster.ami(self.parts(x,keys),y)) if task=='clu' else float(self.raw_ap(x,y,keys).mean()))
                ranges[task]=dict(per_seed={str(s):v for s,v in zip(SEEDS,values)},r=rt_stats.seed_range(values))
            old=self.json(R11/'stage4/picks.json')
            if e=='qwen':p={a:dict(pick=old['cstar' if a=='cstar' else 'plain']) for a in ('cstar','frozen')}
            else:
                r=self.json(MAIN/f'20261102_backbones_sweep/records/stage1_{e}.json')
                p={'cstar':r['artret_picks']['cstar'],'frozen':r['artret_picks']['frozen']}
        teacher=self.json(R11/'stage4/picks.json')['instructed']
        return dict(ranges=ranges,units={t:ranges[t]['r'] for t in ('clu','ret')},picks={**{a:p[a]['pick'] for a in ('cstar','frozen')},'teacher':teacher})

    def get_pick(self,e,d,arm):
        if d=='sem':return self.dep(f'semval:{e}')['picks'][arm]
        if arm=='teacher':
            if d=='art':return self.json(R11/'stage4/picks.json')['instructed']
            if d=='mm':return self.json(MAIN/'20261102_mmart_test/records/stage1_teacher.json')['arms']['teacher']['pick']
            p=self.json(R12/'stage1/picks.json')['affection'];return p['readout']['teacher_'+p['teacher_wording']['retrieval']]
        return self.dep(f'picks:{e}:{d}')[arm]['pick']

    def retrieval(self,e,d,arm,a):
        keys,ys=self.population(d,'test');y=ys[a];pick=self.get_pick(e,d,arm);aps=[]
        if arm=='teacher' and d in ('art','aff'):
            if d=='art':
                path=R11/f'stage5/perquery/instructed__{a}__h{a}__s-.npz'
            else:
                p12=self.json(R12/'stage1/picks.json')['affection']
                path=R12/f'stage2/perquery/affection__teacher_{p12["teacher_wording"]["retrieval"]}__emotion__s-.npz'
            self.track(path)
            # The frozen helper asserts float64 AP, finiteness, and the exact
            # query-row identities against the inherited keyed population.
            return frozen.saved_ap(path,y),pick
        for s in SEEDS if arm=='cstar' else (None,):
            val=self.vector(e,d,'photo' if d=='aff' and arm=='cstar' else arm,'val',a,s,retrieval=True)
            test=self.vector(e,d,'photo' if d=='aff' and arm=='cstar' else arm,'test',a,s,retrieval=True)
            def calc():
                res=ac_stats.arm_retrieval(lambda _:val,lambda _:test,pick,y,workers=8)
                assert np.array_equal(res['rows'],np.flatnonzero(y>=0))
                return res['ap']
            aps.append(self.array('picked_ap',[keys,val,test,pick,y],calc))
        return (np.stack(aps) if arm=='cstar' else aps[0]),pick

    def cells(self,e,d,task=None,smoke=False):
        keys,ys=self.population(d,'test');result=[]
        ranges=self.dep(f'semval:{e}')['ranges'] if d=='sem' else self.dep(f'ranges:{e}:{d}')
        for a,y in ys.items():
            if smoke and a!='emotion':continue
            for t in (('clu','ret') if task is None else (task,)):
                if d=='art':specs=[('SC1','frozen'),('SC3','matched'),('SC6','teacher')] if t=='clu' else [('SC2','frozen'),('SC6','teacher')]
                elif d=='mm':specs=[('SC7','frozen'),('SC9','matched'),('SC10','teacher')] if t=='clu' else [('SC8','frozen'),('SC10','teacher')]
                elif d=='sem':specs=[('SC11','frozen'),('SC13','matched'),('SC14','teacher')] if t=='clu' else [('SC12','frozen'),('SC14','teacher')]
                else:specs=[('SC4' if t=='clu' else 'SC5','frozen'),('SC6','teacher')]
                if smoke:
                    # Smoke covers each bar kind plus one teacher; full SemArt uses all five cells.
                    specs=[specs[0]]+([specs[1]] if d=='art' and t=='clu' else [])+([specs[-1]] if d=='mm' and t=='ret' else [])
                cand=None
                if t=='clu':cand=np.stack([self.parts(self.vector(e,d,'photo' if d=='aff' else 'cstar','test',a,s),keys) for s in SEEDS])
                else:cand,pick=self.retrieval(e,d,'cstar',a)
                for sc,arm in specs:
                    extra={}
                    if t=='clu':
                        if arm=='matched':
                            other=self.matched(e,d)
                            def calc():
                                res=frozen.rb_score.paired_bootstrap({'c':{s:{'r20':cand[s]} for s in SEEDS},'b':other},y,{'c':list(SEEDS),'b':list(SEEDS)},('c','b'),self.n_clu,workers=8)
                                return [res['A_cand'],max(res['A_e'].values()),res['d'],*res['interval']]
                            inputs=[other[s][r] for s in SEEDS for r in ('own','r20')]
                        else:
                            other=self.parts(self.vector(e,d,arm,'test',a),keys)
                            def calc():
                                if d=='aff' or arm=='teacher':
                                    res=ac_cluster.paired(cand,other,y,draws=self.n_clu,workers=8)
                                    return [res['A_cand'],res['A_comp'],res['d'],*res['interval']]
                                res=frozen.rb_score.paired_bootstrap({'c':{s:{'r20':cand[s]} for s in SEEDS},'b':{0:{'r20':other}}},y,{'c':list(SEEDS)},('c','b'),self.n_clu,workers=8,seedless=('b',))
                                return [res['A_cand'],max(res['A_e'].values()),res['d'],*res['interval']]
                            inputs=[other]
                        stats=self.array('clu_cell',[keys,y,cand,sc,self.n_clu]+inputs,calc)
                        extra['n_labelled']=int((y>=0).sum())
                    else:
                        other,other_pick=self.retrieval(e,d,arm,a)
                        def calc():
                            res=rt_stats.paired(cand,other,draws=self.n_ret)
                            return [float(cand.mean()),float(other.mean()),res['d'],*res['interval']]
                        stats=self.array('ret_cell',[keys,y,cand,other,self.n_ret],calc)
                        extra.update(candidate_readout=pick,comparator_readout=other_pick,n_queries=cand.shape[-1])
                    cv,bv,delta,lo,hi=map(float,stats)
                    r=(ranges[t]['r'] if d=='sem' else ranges[a][t]['r']) if arm=='teacher' else None
                    comparator=f'matched_{MATCHED[e]}' if arm=='matched' else f'teacher_{TEACHER[a] if d=="aff" else a}' if arm=='teacher' else f'frozen_{e}'
                    result.append(dict(sc=sc,encoder=e,aspect=a,task=t,data=DATASETS[d],candidate=f'{"photo" if d=="aff" else "cstar"}_{e}',comparator=comparator,candidate_value=cv,comparator_value=bv,d=delta,lo=lo,hi=hi,r=r,outcome=cell_outcome(lo,r),n_draws=self.n_clu if t=='clu' else self.n_ret,seeds=list(SEEDS),n_rows=len(y),**extra))
        return result


def raw_cost(root,track=lambda p:p,prep_meta=None):
    """Rebuild timings from batch arrays; never trust their stored summaries."""
    root=Path(root);passes=read_json(track(root/'probe/passes.json'))
    n=passes['meta']['n_images'];assert n>0
    discarded=passes.get('repeats_discarded',[])
    # Compare actual file identity AND bytes: a discarded file can be copied,
    # hard-linked or renamed into batches under a different name.
    discarded_paths=list((root/'probe/discarded').rglob('*.json'))
    discarded_hashes={file_sha(track(p)) for p in discarded_paths}
    discarded_inodes={(p.stat().st_dev,p.stat().st_ino) for p in discarded_paths}
    def discarded_pass(row):
        for key in ('pass','pass_name','name'):
            if isinstance(row.get(key),str):return row[key]
        filename=row.get('file',row.get('path',''))
        return Path(filename).name.split('__',1)[0] if filename else None
    rates={};table={};used=set()
    for name,entry in passes['passes'].items():
        rr={}
        refs={**entry.get('repeats',{}),**({'e2e':entry['e2e']} if entry.get('e2e') else {})}
        for repeat,ref in refs.items():
            filename=ref['file'];path=root/'probe/batches'/filename
            assert path.resolve().is_relative_to((root/'probe/batches').resolve()),'timed file outside batches'
            stat=path.stat()
            assert (stat.st_dev,stat.st_ino) not in discarded_inodes and file_sha(path) not in discarded_hashes,'discarded repeat timed'
            assert path not in used,'timed batch file reused';used.add(path)
            raw=read_json(track(path))
            assert raw['pass']==name and raw['n_images']==n
            assert raw['batch_size']==32
            # E2E batch files describe the decode loop (warmup_batches=0);
            # model/network repeats separately record three untimed warmups.
            # Rebuild the timed sums without inserting any warmup samples.
            assert raw['warmup_batches'] in ((0,3) if repeat=='e2e' else (3,))
            assert raw['repeat']==(None if repeat=='e2e' else int(repeat))
            sizes,seconds=raw['batch_n'],raw['batch_seconds']
            assert len(sizes)==len(seconds) and sum(sizes)==n
            assert all(0<i<=32 for i in sizes) and all(np.isfinite(s) and s>=0 for s in seconds)
            rr[str(repeat)]=sum(seconds)*1000/n
        rates[name]=rr
        meta={k:entry[k] for k in ('encoder','kind','peak_memory_allocated_mib','peak_memory_reserved_mib','params_loaded','params_image_path','params_network','vision_tower') if k in entry}
        vals=[rr[str(r)] for r in range(3)] if '0' in rr else []
        if vals:
            meta['ms_per_image_mean']=float(np.mean(vals));meta['images_per_second']=1000/meta['ms_per_image_mean']
        if 'e2e' in rr:meta.update(ms_per_image_e2e=rr['e2e'],images_per_second_e2e=1000/rr['e2e'])
        if 'vision_tower' in meta:
            v=dict(meta['vision_tower'])
            v['total_seconds']=sum(v['batch_seconds']);v['visual_seconds']=sum(v['visual_batch_seconds']);v['share']=v['visual_seconds']/v['total_seconds']
            meta['vision_tower']=v
        table[name]=meta
    teachers=['qwen_'+TEACHER[a] for a in ASPECTS]
    teacher=[sum(rates[t][str(r)] for t in teachers) for r in range(3)]
    teacher_e2e=rates['teacher']['e2e'];cells=[];whole={}
    for e in ENCODERS:
        x=[rates[e][str(r)] for r in range(3)];heads=[rates['heads_'+e][str(r)] for r in range(3)]
        c=[x[r]+heads[r] for r in range(3)]
        e2e=rates[e]['e2e']+float(np.mean(heads))
        used_passes={e,'heads_'+e,'teacher',*teachers}
        encoder_discarded=[row for row in discarded if discarded_pass(row) in used_passes]
        cells.append(dict(sc='SC15',encoder=e,cstar_ms=c,teacher_ms=teacher,cstar_ms_max=max(c),teacher_ms_min=min(teacher),
                          cstar_ms_mean=float(np.mean(c)),teacher_ms_mean=float(np.mean(teacher)),cstar_ms_e2e=e2e,teacher_ms_e2e=teacher_e2e,
                          repeats_discarded=len(encoder_discarded),repeats_discarded_log=encoder_discarded,x_ms=x,networks_ms=heads,n_images=n))
        whole[e]=dict(cstar_s=float(np.mean(c))*16384/1000,teacher_s=float(np.mean(teacher))*16384/1000,
                      cstar_e2e_s=e2e*16384/1000,teacher_e2e_s=teacher_e2e*16384/1000,paintings=16384)
    qwen=rates['qwen']['e2e'];rel=abs(qwen/86.3-1)
    gate=dict(pass_=rel<=.2,qwen_default_e2e_ms=qwen,reference_ms=86.3,tolerance=.2,relative_difference=rel,low_ms=86.3*.8,high_ms=86.3*1.2,smoke=n!=2000,n_images=n)
    gate['pass']=gate.pop('pass_')
    if prep_meta is None:
        prep_meta={}
        for name in ('qwen_default',*TEACHER.values()):
            folder=PRE/'mmart'/('qwen3vl_emb_2b' if name=='qwen_default' else 'qwen3vl_emb_2b_aspect_prompts/'+name)
            prep_meta[name]=read_json(track(folder/'meta.json'))
    prep={name:dict(seconds=row['seconds'],paintings=row['n'],ms_per_image=row['seconds']/row['n']*1000) for name,row in prep_meta.items()}
    return dict(cells=cells,gate=gate,table=dict(passes=table,whole_set_16384=whole,prep_log_qwen=prep,n_images=n,passes_per_aspect=dict(cstar=1/3,teacher=1)))


def good_status():
    return dict(gates=dict(i_store={e:dict.fromkeys(ds,'ok') for e,ds in [('dinov2_large',('mmart','affection','semart')),('clip_b32',('mmart','affection')),('clip_b16',('mmart','affection','semart')),('siglip1',('semart',)),('clip_l14',('semart',))]},
            i_art_parity=dict.fromkeys((*NEW,'siglip1','clip_l14'),'ok'),i_semart_parity=dict.fromkeys(('clip_b32','qwen','siglip2'),'ok'),
            i_semart_genre_word='ok',i_aff_l14_parity='ok',ii='ok',iii=dict.fromkeys(('r10','r14'),'ok'),iv='ok',v=dict.fromkeys(NEW,'ok'),vi='ok',vii=dict.fromkeys(('artelingo','affection','mmart'),'ok'),viii='ok'),
            runs={e:dict.fromkeys(('cstar','photo','matched'),'ok') for e in NEW})


def validate_status(status):
    def walk(a,b):
        if isinstance(b,dict):
            assert isinstance(a,dict) and set(a)==set(b),'status keys differ'
            for k in b:walk(a[k],b[k])
        else:assert a in ('ok','failed'),'status must be ok or failed'
    walk(status,good_status())
    for e in NEW:assert status['runs'][e]['cstar']!='failed' or status['runs'][e]['photo']=='failed'


def gate_failed(status,name,derived_gates=None):
    """A failed measurement or any disagreement cannot be cured by status."""
    recorded=status['gates'][name]=='ok'
    if derived_gates is not None and name in derived_gates:
        measured=derived_gates[name]
        assert type(measured) is bool,f'derived gate {name} must be boolean'
        return not measured or measured!=recorded
    return not recorded


def voids(status,e,d,sc=None,task=None,derived_gates=None):
    """The §3.5 table. sc=None asks for the verdict's dependencies."""
    g,r=status['gates'],status['runs'];why=[];new=e in NEW
    def add(bad,name):
        if bad:why.append(name)
    add(new and g['ii']=='failed','gates.ii')
    add(new and g['iii']['r10']=='failed','gates.iii.r10')
    add(new and g['iii']['r14']=='failed','gates.iii.r14')
    if d=='cost':
        add(gate_failed(status,'viii',derived_gates),'gates.viii')
        add(new and r[e]['cstar']=='failed',f'runs.{e}.cstar')
        return why
    store_set=dict(art='artelingo',aff='affection',mm='mmart',sem='semart')[d]
    stores=g['i_store'].get(e,{})
    add(stores.get(store_set)=='failed',f'gates.i_store.{e}.{store_set}')
    add(store_set in stores and g['i_art_parity'].get(e)=='failed',f'gates.i_art_parity.{e}')
    if d=='sem':
        for enc,s in g['i_semart_parity'].items():add(s=='failed',f'gates.i_semart_parity.{enc}')
        add(gate_failed(status,'vi',derived_gates),'gates.vi')
        add(sc=='SC14' and g['i_semart_genre_word']=='failed','gates.i_semart_genre_word')
        add(not new and sc=='SC14' and task=='ret' and g['vii']['artelingo']=='failed','gates.vii.artelingo')
    add(d=='aff' and g['i_aff_l14_parity']=='failed','gates.i_aff_l14_parity')
    add(d=='sem' and e=='qwen' and g['iii']['r10']=='failed','gates.iii.r10')
    add(d=='sem' and e in ENCODERS[1:4] and g['iii']['r14']=='failed','gates.iii.r14')
    add(g['iv']=='failed','gates.iv')
    if new:
        add(r[e]['cstar']=='failed',f'runs.{e}.cstar')
        add(d=='aff' and r[e]['photo']=='failed',f'runs.{e}.photo')
        if d in ('art','mm','sem') and (sc is None or sc in ('SC3','SC9','SC13')):
            add(g['v'][e]=='failed',f'gates.v.{e}')
            add(r[e]['matched']=='failed',f'runs.{e}.matched')
        route='artelingo' if d in ('art','sem') else 'affection' if d=='aff' else 'mmart'
        add((sc is None or task=='ret') and g['vii'][route]=='failed',f'gates.vii.{route}')
    return why


def teacher_claim(e,d,task,outcome,void=False):
    if void:return 'inconclusive'
    if outcome=='WIN' and d=='sem' and task=='ret' and e in ENCODERS[:4]:return 'better than'
    return 'level with or better than' if outcome in ('WIN','LEVEL') else None


def expected_ids():
    rows=[]
    for e in NEW:
        for d,offset in [('art',0),('mm',6)]:
            for a in ASPECTS:
                for sc,t in [(1,'clu'),(2,'ret'),(3,'clu'),(6 if d=='art' else 4,'clu'),(6 if d=='art' else 4,'ret')]:
                    num=sc+offset if d=='mm' else sc
                    rows.append((f'SC{num}',e,a,t,DATASETS[d]))
        for sc,t in [('SC4','clu'),('SC5','ret'),('SC6','clu'),('SC6','ret')]:rows.append((sc,e,'emotion',t,DATASETS['aff']))
    for e in ENCODERS:
        for sc,t in [('SC11','clu'),('SC12','ret'),('SC13','clu'),('SC14','clu'),('SC14','ret')]:rows.append((sc,e,'genre',t,DATASETS['sem']))
    assert len(rows)==137 and len(set(rows))==137
    return set(rows)


def cell_key(c):
    return tuple(c[k] for k in ('sc','encoder','aspect','task','data'))


def index_cells(cells):
    result={}
    for c in cells:
        k=cell_key(c);assert k not in result,f'duplicate cell: {k}';result[k]=c
    return result


def decisions(cells,cost,status,partial=False,counts=None,derived_gates=None):
    validate_status(status);idx=index_cells(cells);costidx={c['encoder']:c for c in cost}
    assert len(costidx)==len(cost),'duplicate cost encoder'
    reverse={v:k for k,v in DATASETS.items()};verdicts={};claims=[];decorated=[];void_table={}
    for sc,e,a,t,data in expected_ids()-set(idx):
        assert partial or voids(status,e,reverse[data],sc,t,derived_gates),f'missing live cell: {sc} {e} {a} {t}'
    for c in cells:
        d=reverse[c['data']];why=voids(status,c['encoder'],d,c['sc'],c['task'],derived_gates);outcome=cell_outcome(c['lo'],c['r'])
        row={**c,'outcome':outcome,'status':'inconclusive' if why else 'decided','void':why};decorated.append(row)
        if c['sc'] in ('SC6','SC10','SC14'):
            claims.append({**{k:c[k] for k in IDENTITY},'outcome':outcome,'status':row['status'],'void':why,'claim':teacher_claim(c['encoder'],d,c['task'],outcome,bool(why))})
    for e in ENCODERS:
        verdicts[e]={};void_table[e]={}
        for d in (('art','aff','mm','sem','cost') if e in NEW else ('sem','cost')):
            why=voids(status,e,d,derived_gates=derived_gates);void_table[e][d]=why
            specs=dict(art=(('SC1','SC2','SC3'),9,'SC3'),aff=(('SC4','SC5'),2,None),mm=(('SC7','SC8','SC9'),9,'SC9'),sem=(('SC11','SC12','SC13'),3,'SC13'))
            if why:v='inconclusive'
            elif d=='cost':
                assert partial or e in costidx,f'missing SC15 {e}'
                v='pending' if e not in costidx else 'holds' if costidx[e]['cstar_ms_max']<costidx[e]['teacher_ms_min'] else 'fails'
            else:
                scs,n,matched=specs[d];rows=[c for c in decorated if c['encoder']==e and c['data']==DATASETS[d] and c['sc'] in scs]
                if len(rows)!=n:
                    assert partial,'missing required cells';v='pending'
                elif all(c['outcome']=='WIN' for c in rows):v='holds'
                elif matched and all(c['outcome']=='WIN' for c in rows if c['sc']!=matched):v='qualified'
                else:v='fails'
            verdicts[e][d]=v
    decorated_cost=[]
    for c in cost:
        why=voids(status,c['encoder'],'cost','SC15',derived_gates=derived_gates)
        decorated_cost.append({**c,'status':'inconclusive' if why else 'decided','void':why})
    return dict(verdicts=verdicts,voids=void_table,claims=sorted(claims,key=cell_key),cells=sorted(decorated,key=cell_key),cost=decorated_cost,counts=counts or {},derived_gates=derived_gates or {})


def run_section(w,name,force=False):
    assert name in FULL_SECTIONS or name in SMOKE_SECTIONS,f'unknown section {name}'
    p=w.out/'sections'/f'{name}.json'
    if p.exists() and not force:
        w.validate(read_json(p));print(f'skip {name}',flush=True);return
    w.inputs={};start=time.monotonic();bits=name.split(':')
    # Labels and the frozen split identities are inputs too, including for
    # resumability of a complete section (not just its array caches).
    data=bits[2] if len(bits)>2 else 'art' if bits[0] in ('set','semval') else None
    if data in ('art','sem'):
        for path in (frozen.rs_vh.ANNOTATIONS,*frozen.rs_vh.GENRE_CSVS,
                     PRE/'artelingo_full/qwen3vl_emb_2b/index.json',
                     MAIN/'20261020_claim_test/outputs/held/held_paintings.json'):
            w.track(path)
    elif data=='aff':w.track(ac_data.ac_paths.ANNOTATIONS)
    with threadpool_limits(w.threads):
        if name=='set':result=w.semset()
        elif name=='cost':result=raw_cost(w.root,w.track)
        elif bits[0]=='ranges':result=w.ranges(bits[1],bits[2])
        elif bits[0]=='picks':result=w.picks(bits[1],bits[2])
        elif bits[0]=='semval':result=w.semval(bits[1])
        else:result=w.cells(bits[1],bits[2],bits[3] if len(bits)>3 else None,bits[0]=='smoke')
    atomic_json(p,dict(name=name,complete=True,root=str(w.root),draws_clu=w.n_clu,draws_ret=w.n_ret,kmeans_threads=w.threads,
                      source_sha256=source_identity(),code_sha256=file_sha(__file__),inputs=w.inputs,result=result,seconds=time.monotonic()-start))
    print(f'complete {name} {time.monotonic()-start:.1f}s',flush=True)


def assemble(out=DEFAULT_OUT,require_full=True,sections=None):
    out=check_output(out);paths=sorted((out/'sections').glob('*.json'));docs={}
    for path in paths:
        if sections is not None and path.stem not in sections:continue
        d=read_json(path);assert d['name']==path.stem and d['complete'],'incomplete section'
        docs[d['name']]=d
    if sections is not None and set(sections)-set(docs):
        raise ValueError('missing selected sections: '+', '.join(sorted(set(sections)-set(docs))))
    missing=set(FULL_SECTIONS)-set(docs)
    if require_full and missing:raise ValueError('missing sections: '+', '.join(sorted(missing)))
    if not docs:raise ValueError('no sections')
    configs={(d['root'],d['draws_clu'],d['draws_ret'],d['kmeans_threads'],d['source_sha256']) for d in docs.values()}
    if len(configs)!=1:raise ValueError('section roots, draw counts, threads or sources differ')
    root,nclu,nret,threads,source=configs.pop();cells=[];cost=[];inputs={}
    for name,d in docs.items():
        for p,h in d['inputs'].items():
            if p in inputs and inputs[p]!=h:raise ValueError(f'conflicting input bytes: {p}')
            inputs[p]=h
        if name.startswith(('cells:','smoke:')):cells.extend(d['result'])
        if name=='cost':cost=d['result']['cells']
    # A full cell section supersedes a smoke section for that same path.
    cells=[]
    for name,d in docs.items():
        if name.startswith('smoke:') and name.replace('smoke:','cells:',1) in docs:continue
        if name.startswith(('cells:','smoke:')):cells.extend(d['result'])
    index_cells(cells)
    full=not missing and set(index_cells(cells))==expected_ids() and {c['encoder'] for c in cost}==set(ENCODERS)
    status_path=RECORDS/'status.json';status=read_json(status_path);inputs[str(status_path)]=file_sha(status_path)
    setdoc=docs.get('set',{}).get('result',{})
    mmart_keys=None
    if full or any(c['data']=='mmart_test' for c in cells):
        work=Work(root,out,nclu,nret,threads)
        mmart_keys,_=work.population('mm','test')
        inputs.update(work.inputs)
    derived_gates={}
    for section,gate in (('set','vi'),('cost','viii')):
        if section in docs:
            value=docs[section]['result'].get('gate',{}).get('pass')
            assert type(value) is bool,f'missing or nonboolean derived gate {gate}'
            derived_gates[gate]=value
    counts=dict(mmart_n=len(mmart_keys) if mmart_keys is not None else None,semart_n=setdoc.get('n_set'),semart_artists=setdoc.get('artists_set'))
    decided=decisions(cells,cost,status,partial=not full,counts=counts,derived_gates=derived_gates)
    result=dict(root=root,sections=sorted(docs),scope='full' if full else 'subset:'+','.join(sorted(docs)),
                draws_clu=nclu,draws_ret=nret,kmeans_threads=threads,source_sha256=source,inputs=inputs,
                results={n:d['result'] for n,d in docs.items()},cells=cells,cost=cost,status=status,decisions=decided,
                mmart_test_keys=mmart_keys,derived_gates=derived_gates)
    if require_full and not full:raise ValueError('full section names do not cover all 137 cells and 7 SC15 cells')
    return result


def merge(out=DEFAULT_OUT):
    doc=assemble(out,True);atomic_json(Path(out)/'derived.json',doc);return doc


class Comparison:
    def __init__(self):self.items=[]
    def add(self,name,a,b,numeric=None,section=None):
        discrete=numeric is None and isinstance(a,int) and not isinstance(a,bool)
        if numeric is None:numeric=isinstance(a,(int,float)) and not isinstance(a,bool)
        diff=None
        if numeric:
            ok=isinstance(b,(int,float)) and not isinstance(b,bool) and np.isfinite(a) and np.isfinite(b)
            if ok:diff=abs(a-b);ok=diff<=TOL and (not discrete or type(a)==type(b) and a==b)
        else:ok=type(a)==type(b) and a==b
        self.items.append(dict(name=name,derived=a,recorded=b,abs_diff=diff,pass_=bool(ok),section=section))
        self.items[-1]['pass']=self.items[-1].pop('pass_')
    def tree(self,name,a,b,section=None):
        if isinstance(a,dict):
            self.add(name+'/keys',sorted(a),sorted(b) if isinstance(b,dict) else None,False,section)
            for k,v in a.items():self.tree(name+'/'+str(k),v,b.get(k) if isinstance(b,dict) else None,section)
        elif isinstance(a,list):
            self.add(name+'/length',len(a),len(b) if isinstance(b,list) else None,False,section)
            for i,v in enumerate(a):self.tree(name+f'/{i}',v,b[i] if isinstance(b,list) and i<len(b) else None,section)
        else:self.add(name,a,b,section=section)
    def report(self,scope,doc=None):
        failed=sum(not x['pass'] for x in self.items);max_abs=max([x['abs_diff'] for x in self.items if x['abs_diff'] is not None]+[0.])
        sections={}
        for s in sorted({x['section'] for x in self.items if x['section']}):
            rows=[x for x in self.items if x['section']==s]
            sections[s]=dict(items=len(rows),failed=sum(not x['pass'] for x in rows),max_abs=max([x['abs_diff'] for x in rows if x['abs_diff'] is not None]+[0.]))
        return dict(pass_=failed==0,items=self.items,n_items=len(self.items),failed=failed,max_abs=max_abs,scope=scope,per_section=sections,
                    draws_clu=(doc or {}).get('draws_clu'),draws_ret=(doc or {}).get('draws_ret'),kmeans_threads=(doc or {}).get('kmeans_threads'))


def summary(report,scope=None):
    scope=scope or ('subset' if report['scope'].startswith('subset') else report['scope'])
    return f'C13 {"PASS" if report["pass"] else "FAIL"} items={report["n_items"]} failed={report["failed"]} max_abs={report["max_abs"]:.12g} scope={scope}'


def result_paths(root,records):
    """Exactly c13_check's path set and key spelling, including raw discards."""
    root,records=Path(root),Path(records)
    paths={str(p.relative_to(root)):p for d in ('stage2','stage3') for p in (root/d).glob('cells_*.json') if not p.name.endswith('.pending.json')}
    paths.update({str(p.relative_to(root)):p for p in (root/'probe').rglob('*.json')})
    paths.update({f'records/{p.name}':p for p in [records/'status.json',records/'semart_set.json',*records.glob('stage1_*.json')]})
    return paths


def record_refusal(doc,report,round_cells):
    if doc['scope']!='full':return 'subset scope cannot certify C13'
    if not report['pass']:return 'comparison failed'
    if doc.get('kmeans_threads')!=8:return 'record requires kmeans_threads 8'
    if (doc.get('draws_clu'),doc.get('draws_ret'))!=(2000,10000):return 'record requires draws 2000 and 10000'
    sem=doc.get('results',{}).get('set',{})
    if sem.get('n_set')!=11159 or not sem.get('gate',{}).get('pass'):return 'SemArt set gate did not pass'
    status=doc.get('status',{})
    for section,gate in (('set','vi'),('cost','viii')):
        measured=doc.get('results',{}).get(section,{}).get('gate',{}).get('pass')
        recorded=status.get('gates',{}).get(gate)
        if measured is not True or recorded!='ok':return f'derived gate ({gate}) failed, missing, or disagrees with status'
    if len(doc.get('cost',[]))!=7 or any(c.get('n_images')!=2000 for c in doc['cost']):return 'record requires all seven 2000-image probe cells'
    if set(index_cells(round_cells))!=expected_ids():return 'round cell coverage is incomplete'
    for c in round_cells:
        if c.get('n_draws')!=(2000 if c['task']=='clu' else 10000):return 'round draw counts differ'
        if c.get('n_rows')!=dict(artelingo_held=12281,affection_test=8501,mmart_test=16384,semart_new=11159)[c['data']]:return 'round population counts differ'
    return None


def compare(root=DEFAULT_ROOT,records=RECORDS,out=DEFAULT_OUT,write_record=False,sections=None):
    wording=read_json(R12/'stage1/picks.json')['affection']['teacher_wording'];assert wording['clustering']==wording['retrieval'];r12_teacher='teacher_'+wording['retrieval']
    root,records,out=Path(root).resolve(),Path(records).resolve(),check_output(out)
    doc=assemble(out,False) if sections is None else assemble(out,False,sections)
    atomic_json(out/('derived.json' if doc['scope']=='full' else 'derived_subset.json'),doc)
    cmp=Comparison();hashes={};read_bytes={};loaded={}
    def load(path):
        path=Path(path)
        if path in loaded:return loaded[path]
        if not path.exists():return None
        data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
        name='records/'+path.name if path.parent==records else str(path.relative_to(root))
        hashes[name]=h;read_bytes[path]=h
        loaded[path]=json.loads(data)
        return loaded[path]
    cmp.add('root',doc['root'],str(root),False,'integrity')
    cmp.add('source_sha256',doc['source_sha256'],source_identity(),False,'integrity')
    for p,h in doc['inputs'].items():cmp.add('input/'+p,h,file_sha(p),False,'integrity')
    cmp.tree('status',doc['status'],load(records/'status.json'),'status')
    for gate,measured in doc.get('derived_gates',{}).items():
        cmp.add('gates/'+gate+'/measurement_matches_status',measured,doc['status']['gates'][gate]=='ok',False,'gates')
    for name,result in doc['results'].items():
        bits=name.split(':')
        if name=='set':
            actual=load(records/'semart_set.json') or {}
            for k,v in result.items():
                if k in ('keys','gate_keys'):continue
                if k=='inner_pairs_ge_0.95':
                    for sub,value in v.items():cmp.add('set/'+k+'/'+sub,value,actual.get(k,{}).get(sub),section=name)
                elif k=='gate':
                    for sub,value in v.items():cmp.tree('set/gate/'+sub,value,actual.get('gate',{}).get(sub),name)
                else:cmp.tree('set/'+k,v,actual.get(k),name)
        elif bits[0] in ('ranges','picks'):
            _,e,d=bits;actual=load(records/f'stage1_{e}.json') or {}
            if bits[0]=='ranges':
                for a,tasks in result.items():
                    for task,values in tasks.items():
                        if d=='aff':
                            metric='map' if task=='ret' else task
                            raw=actual.get('affection',{}).get('ranges_doc',{}).get(metric,{})
                            recorded=dict(r=raw.get('r'),per_seed={str(i):v for i,v in enumerate(raw.get('per_seed',[]))})
                        else:
                            block=(actual.get('mmart',{}).get('clu_ranges' if task=='clu' else 'ret_ranges',{}) if d=='mm' else actual.get('artclu' if task=='clu' else 'artret_ranges',{}))
                            recorded=dict(r=block.get('ranges',{}).get(a),per_seed=block.get('per_seed',{}).get(a))
                        cmp.tree(name+f'/{a}/{task}',values,recorded,name)
            else:
                for arm,values in result.items():
                    if d=='art':block=actual.get('artret_picks',{}).get(arm,{})
                    elif d=='mm':block=actual.get('mmart',{}).get('picks',{}).get('arms',{}).get(arm,{})
                    else:block=actual.get('affection',{}).get('photo_readout' if arm=='cstar' else 'frozen',{})
                    for k in ('pick','table'):
                        cmp.tree(name+'/'+arm+'/'+k,values[k],block.get('scores' if d=='aff' and k=='table' else k),name)
                    if d!='aff' and arm=='cstar':
                        cmp.tree(name+'/'+arm+'/ranges',values['ranges'],block.get('ranges'),name)
                        cmp.tree(name+'/'+arm+'/dropped',values['dropped'],block.get('dropped'),name)
        elif bits[0]=='semval':
            block=(load(records/'stage1_semart.json') or {}).get(bits[1],{})
            cmp.tree(name+'/units',result['units'],block.get('units'),name)
            cmp.tree(name+'/picks',result['picks'],block.get('picks'),name)
        elif name=='cost':
            actual=load(root/'stage3/cells_cost.json')
            actual={c['encoder']:c for c in actual or []}
            for c in result['cells']:cmp.tree('cost/'+c['encoder'],c,actual.get(c['encoder']),name)
            cmp.tree('cost/table',result['table'],load(root/'stage3/cost_table.json'),name)
            cmp.tree('cost/gate',result['gate'],load(root/'stage0/gate_viii.json'),name)
    round_cells=[]
    parts=('artelingo_clu','artelingo_ret','affection','mmart_clu','mmart_ret','semart_clu','semart_ret')
    if doc['scope']=='full':
        present=sorted(p.name for p in (root/'stage2').glob('cells_*.json') if not p.name.endswith('.pending.json'))
        cmp.add('coverage/stage2_files',sorted('cells_'+part+'.json' for part in parts),present,False,'coverage')
        present=sorted(p.name for p in (root/'stage3').glob('cells_*.json') if not p.name.endswith('.pending.json'))
        cmp.add('coverage/stage3_files',['cells_cost.json'],present,False,'coverage')
    for part in parts:
        rows=load(root/f'stage2/cells_{part}.json')
        if isinstance(rows,list):round_cells.extend(rows)
        elif doc['scope']=='full':cmp.add('missing/stage2/'+part,True,False,False,'coverage')
    try:actual=index_cells(round_cells)
    except AssertionError:
        cmp.add('duplicate round cell',False,True,False,'coverage');actual={}
    if doc['scope']=='full':cmp.add('expected cells',sorted(expected_ids()),sorted(actual),False,'coverage')
    for c in doc['cells']:
        k=cell_key(c);other=actual.get(k,{})
        section=next((n for n,r in doc['results'].items() if n.startswith(('cells:','smoke:')) and any(cell_key(x)==k for x in r)),'cells')
        prefix='/'.join(k)
        for field in ('candidate_value','comparator_value','d','lo','hi','r','n_draws','seeds','n_rows','n_labelled','candidate_readout','comparator_readout','n_queries'):
            # r14's Affection cell format has no read-out/query annotations;
            # its complete picks are compared in the stage 1 record above.
            if c['data']=='affection_test' and field in ('candidate_readout','comparator_readout','n_queries','n_labelled') and field not in other:
                continue
            if field in c:cmp.tree(prefix+'/'+field,c[field],other.get(field),section)
        # Matched ArtELingo uses its historical arm name without the prefix.
        for field in ('candidate','comparator'):
            rec=other.get(field)
            if field=='comparator' and rec==MATCHED[c['encoder']]:rec='matched_'+rec
            # The round names Affection's teacher by r12's wording (spec §3.2); C13 by the view, as r14 did.
            if field=='comparator' and c['sc']=='SC6' and c['data']=='affection_test' and rec==r12_teacher:rec='teacher_emotion_focus'
            cmp.add(prefix+'/'+field,c[field],rec,False,section)
        if 'lo' in other:cmp.add(prefix+'/outcome',c['outcome'],cell_outcome(other['lo'],other['r']),False,section)
        else:cmp.add(prefix+'/outcome',c['outcome'],None,False,section)
    # Bind certification to exactly the bytes read, including all raw probe JSONs.
    for name,p in result_paths(root,records).items():
        h=file_sha(p)
        if p in read_bytes:cmp.add('stable/'+name,read_bytes[p],h,False,'integrity')
        hashes[name]=h
    report=cmp.report(doc['scope'],doc);report['pass']=report.pop('pass_');report['sha256']=hashes
    refusal=record_refusal(doc,report,round_cells) if write_record else None
    if write_record:
        report['record']='refused: '+refusal if refusal else 'written'
        p=records/'c13.json'
        if records!=RECORDS:
            check_output(p)
        if refusal:
            # The job forbids deletion: invalidate any earlier passing certificate.
            if p.exists() and read_json(p).get('pass') is True:
                p.write_text(json.dumps(dict(pass_=False,reason=refusal,remove_requested=str(p))).replace('"pass_"','"pass"')+'\n')
        else:
            # The item list stays in <out>/compare.json; the committed record is the summary and the hashes.
            certificate={**{k:v for k,v in report.items() if k!='items'},'sha256':{name:hashes[name] for name in result_paths(root,records)}}
            p.write_text(json.dumps(certificate,indent=1)+'\n')
    atomic_json(out/'compare.json',report);print(summary(report),flush=True)
    if refusal:raise ValueError(refusal)
    return report


def sentence_number_matches(token,value):
    """Compare the printed precision, including optional thousands separators."""
    if value is None or not isinstance(value,(int,float)) or not np.isfinite(value):return False
    decimals=len(token.rsplit('.',1)[1]) if '.' in token else 0
    return token==format(value,(',' if ',' in token else '')+f'.{decimals}f')


# pp_paths.ENCODER_NAME's names that hold digits: those digits are not numbers of the sentence.
DISPLAY_NAMES=('Qwen3-VL-Embedding-2B','SigLIP 2','SigLIP 1','CLIP L/14','CLIP B/32','CLIP B/16')
# §6's licensed cost sentence: its numbers <a>, <b>, <c>, <d> in slot order, so a swap cannot pass.
COST_SLOTS=('cstar_ms_mean','teacher_ms_mean','cstar_ms_e2e','teacher_ms_e2e')
COST_SENTENCE=re.compile(r'embeds a painting for emotion, style and genre in ([\d,.]+) ms\b.*?against ([\d,.]+) ms\b.*?\(([\d,.]+) and ([\d,.]+) ms with decoding\)')


def strip_display_names(text):
    for name in DISPLAY_NAMES:text=text.replace(name,'')
    return text


def check_outcomes(root=DEFAULT_ROOT,out=DEFAULT_OUT,records=RECORDS):
    """Compare later stage 4 by cell identity, set/encoder and teacher identity.

    Sentence prose is not a numerical input: its numbers are checked against
    independently derived cell, population and timing values. Unknown numeric
    fields fail closed, so additions to the outcome layout cannot be skipped.
    """
    wording=read_json(R12/'stage1/picks.json')['affection']['teacher_wording'];assert wording['clustering']==wording['retrieval'];r12_teacher='teacher_'+wording['retrieval']
    root,out=Path(root),check_output(out);doc=read_json(out/'derived.json')
    assert doc['scope']=='full','check-outcomes requires a full derive'
    path=root/'stage4/outcomes.json';data=path.read_bytes();actual=json.loads(data)
    # Stage 4 decides from the committed status record and does not copy it into outcomes.json.
    recorded_status=actual['status'] if 'status' in actual else read_json(Path(records)/'status.json')
    expected=doc['decisions'];cmp=Comparison();cmp.tree('status',doc['status'],recorded_status,'outcomes')
    seen_cells=set();seen_cost=set();seen_verdicts=set();seen_claims=set()
    idx=index_cells(expected['cells']);claims={cell_key(c):c for c in expected['claims']};cost={c['encoder']:c for c in doc['cost']}
    aliases=dict(artelingo='art',paintings='art',artelingo_held='art',photos='aff',affection='aff',affection_test='aff',mmart='mm',mmart_test='mm',semart='sem',semart_new='sem',cost='cost')
    def descend(node,parts=(),context_e=None,context_d=None):
        label='/'.join(map(str,parts));e=next((str(p) for p in parts if str(p) in ENCODERS),context_e)
        d=next((aliases[str(p)] for p in reversed(parts) if str(p) in aliases),context_d)
        if isinstance(node,dict):
            # outcomes.json files claims per encoder, so a claim row has no encoder field of its own.
            if 'encoder' not in node and e in ENCODERS and all(k in node for k in ('sc','aspect','task','data')):node={**node,'encoder':e}
            if all(k in node for k in ('sc','encoder','aspect','task','data')):
                key=cell_key(node);target=idx.get(key)
                if 'claim' in node and 'candidate_value' not in node:
                    target=claims.get(key);seen_claims.add(key)
                    if target is None:cmp.add(label+'/expected claim',True,False,False,'outcomes');return
                    for k in ('outcome','claim','status'):cmp.add(label+'/'+k,target[k],node.get(k),False,'outcomes')
                    cmp.add(label+'/void',bool(target['void']),bool(node.get('void')),False,'outcomes')
                    if isinstance(node.get('sentence'),str):descend(node['sentence'],parts+('sentence',),e,d)
                else:
                    seen_cells.add(key)
                    if target is None:cmp.add(label+'/expected cell',True,False,False,'outcomes');return
                    for k,v in target.items():
                        if k=='void':cmp.add(label+'/void',bool(v),bool(node.get(k)),False,'outcomes')
                        elif k in node or k in ('candidate_value','comparator_value','d','lo','hi','r','outcome','status'):
                            recorded=node.get(k)
                            if k=='comparator' and recorded==MATCHED[node['encoder']]:recorded='matched_'+recorded
                            if k=='comparator' and node['sc']=='SC6' and node['data']=='affection_test' and recorded==r12_teacher:recorded='teacher_emotion_focus'
                            cmp.tree(label+'/'+k,v,recorded,'outcomes')
                    if 'claim' in node and key in claims:
                        seen_claims.add(key);cmp.add(label+'/claim',claims[key]['claim'],node['claim'],False,'outcomes')
                return
            if node.get('sc')=='SC15' and node.get('encoder') in ENCODERS:
                enc=node['encoder'];seen_cost.add(enc)
                for k,v in cost[enc].items():cmp.tree(label+'/'+k,v,node.get(k),'outcomes')
                why=expected['voids'][enc]['cost'];v=expected['verdicts'][enc]['cost']
                cmp.add(label+'/void',bool(why),bool(node.get('void')),False,'outcomes')
                if 'status' in node:cmp.add(label+'/status','inconclusive' if why else 'decided',node['status'],False,'outcomes')
                if 'verdict' in node:
                    seen_verdicts.add((enc,'cost'));cmp.add(label+'/verdict',v,node['verdict'],False,'outcomes')
                return
            enc=node.get('encoder',e)
            dataset=node.get('set',node.get('data'))
            ds=aliases.get(dataset,d)
            if 'verdict' in node and enc in ENCODERS and ds:
                seen_verdicts.add((enc,ds));cmp.add(label+'/verdict',expected['verdicts'].get(enc,{}).get(ds),node['verdict'],False,'outcomes')
                cmp.add(label+'/void',bool(expected['voids'][enc][ds]),bool(node.get('void')),False,'outcomes')
            for k,v in node.items():
                if k in ('verdict','void') and enc in ENCODERS and ds:continue
                descend(v,parts+(k,),enc,ds)
        elif isinstance(node,list):
            for i,v in enumerate(node):descend(v,parts+(i,),e,d)
        elif isinstance(node,str):
            if node in ('holds','qualified','fails','inconclusive') and e and d:
                seen_verdicts.add((e,d));cmp.add(label,expected['verdicts'][e].get(d),node,False,'outcomes')
            # Licensed sentence counts are protocol constants / derived populations.
            sentence_path=any('sentence' in str(p) or str(p) in ('licensed','limit') for p in parts)
            if (len(node)>100 or sentence_path) and re.search(r'\d',node):
                population=re.search(r'([\d,]+) paintings by ([\d,]+) artists',node)
                if population:
                    cmp.add(label+'/paintings',expected['counts']['semart_n'],int(population[1].replace(',','')),section='outcomes')
                    cmp.add(label+'/artists',expected['counts']['semart_artists'],int(population[2].replace(',','')),section='outcomes')
                numbers=re.findall(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?',strip_display_names(node))
                permitted=[20,10,5,32,3090,3,1,.95,12281,9949,8501,doc['decisions']['counts'].get('mmart_n'),doc['decisions']['counts']['semart_n'],doc['decisions']['counts']['semart_artists']]
                if e and e in cost:permitted += [cost[e][k] for k in ('cstar_ms_mean','teacher_ms_mean','cstar_ms_e2e','teacher_ms_e2e','cstar_ms_max','teacher_ms_min') if k in cost[e]]
                if 'ms with decoding' in node:
                    slots=COST_SENTENCE.search(node)
                    cmp.add(label+'/cost_slot/parsed',True,bool(slots and e and e in cost),False,'outcomes')
                    if slots and e and e in cost:
                        for k,token in zip(COST_SLOTS,slots.groups()):
                            cmp.add(label+'/cost_slot/'+k,True,sentence_number_matches(token,cost[e].get(k)),False,'outcomes')
                for i,token in enumerate(numbers):
                    match=next((v for v in permitted if sentence_number_matches(token,v)),None)
                    expected_token=None if match is None else format(match,(',' if ',' in token else '')+f'.{len(token.rsplit(".",1)[1]) if "." in token else 0}f')
                    cmp.add(label+f'/sentence_number/{i}',expected_token,token,False,'outcomes')
        elif isinstance(node,(int,float)) and not isinstance(node,bool):
            # Context-free numbers must equal a derived number with the same field.
            k=str(parts[-1]);candidates=[]
            if len(parts)>1 and parts[-2]=='seeds' and isinstance(parts[-1],int):
                cmp.add(label,SEEDS[parts[-1]] if parts[-1]<5 else None,node,section='outcomes');return
            if k in expected['counts']:candidates.append(expected['counts'][k])
            if e and e in cost and k in cost[e]:candidates.append(cost[e][k])
            if k in ('mmart_n','semart_n','semart_artists'):candidates.append(expected['counts'].get(k))
            if k in ('n_rows','n','paintings','artists','n_images','n_seeds','seeds','clusters','n_draws'):
                candidates += [12281,9949,8501,16384,11159,2717,2000,10000,5,20]
            cmp.add(label,True,any(v is not None and isinstance(v,(int,float)) and abs(node-v)<=TOL for v in candidates),False,'outcomes')
    # Status was compared exactly already; prevent its prose leaves being counted.
    descend({k:v for k,v in actual.items() if k!='status'})
    cmp.add('coverage/cells',sorted(idx),sorted(seen_cells),False,'coverage')
    cmp.add('coverage/cost',sorted(cost),sorted(seen_cost),False,'coverage')
    cmp.add('coverage/claims',sorted(claims),sorted(seen_claims),False,'coverage')
    wanted={(e,d) for e,sets in expected['verdicts'].items() for d in sets}
    cmp.add('coverage/verdicts',sorted(wanted),sorted(seen_verdicts),False,'coverage')
    report=cmp.report('outcomes',doc);report['pass']=report.pop('pass_');report['sha256']={'stage4/outcomes.json':hashlib.sha256(data).hexdigest()}
    atomic_json(out/'check_outcomes.json',report);print(summary(report),flush=True);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('sections')
    for name in ('derive','merge','compare','check-outcomes'):
        q=sub.add_parser(name);q.add_argument('--out',type=Path,default=DEFAULT_OUT)
        if name!='merge':q.add_argument('--root',type=Path,default=DEFAULT_ROOT)
        if name=='derive':
            q.add_argument('--draws-clu',type=int,default=2000);q.add_argument('--draws-ret',type=int,default=10000)
            q.add_argument('--kmeans-threads',type=int,default=8);q.add_argument('--sections',nargs='+',required=True);q.add_argument('--force',action='store_true')
        if name=='compare':
            q.add_argument('--records',type=Path,default=RECORDS);q.add_argument('--write-record',action='store_true')
            q.add_argument('--sections',nargs='+',help='Compare only these derived sections; preserve the merged full file.')
    args=p.parse_args(argv)
    try:
        if args.cmd=='sections':
            for name,minutes in schedule():print(f'{name}\t{minutes:g}m')
        elif args.cmd=='derive':
            assert args.draws_clu>0 and args.draws_ret>0
            for name in args.sections:run_section(Work(args.root,args.out,args.draws_clu,args.draws_ret,args.kmeans_threads),name,args.force)
        elif args.cmd=='merge':merge(args.out)
        elif args.cmd=='compare':return 0 if compare(args.root,args.records,args.out,args.write_record,sections=args.sections)['pass'] else 1
        else:return 0 if check_outcomes(args.root,args.out)['pass'] else 1
    except (AssertionError,ValueError,FileNotFoundError) as exc:
        print(f'C13 refused: {exc}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
