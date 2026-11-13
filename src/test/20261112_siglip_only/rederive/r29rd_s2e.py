#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent C13 re-derivation: r29 E2/E3 Table 6 and SC9.

Only held --picks-committed can open stage-2 held exports. Replay exclusively
uses r14 frozen models. No older scoring module is imported. Numerics are shared
with the explicitly permitted independent r29rd SC1/SC2/C1 core, not our targets.
"""
import os
import sys
sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
# C captures the controller's CUDA visibility before our earlier CPU-only imports.
import r29rd_c1_core as C
import r29rd_sc1 as rd
import r29rd_sc2 as S
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import sklearn
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[1]
SPEC = Path('/project/CoSiR/docs/superpowers/specs')
R29 = Path('/data/SSD2/cosir_r29/20261112_siglip_only/cstar')
R14 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R11 = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
REC14 = TEST / '20261102_backbones_sweep/records'
REC17 = TEST / '20261103_new_scores/records'
VALPICK = TEST / '20261112_siglip_main/records/valpick.json'
ENCODERS = ('siglip2', 'clip_l14')
ASPECTS = rd.ASPECTS
SEEDS = tuple(range(5))
ARMS = dict(siglip2='siglip2_prefix', clip_l14='clipl14_prefix')
STORES = dict(siglip2='siglip2_so400m', clip_l14='clip_l14')
INST = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
HELD_SHA = 'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876'
INPUTS = {}
ALLOW_HELD = False
NOTES = [
    'D18 protocols: AMI uses r17 new-scores §§3.3–3.4/ns_c13; mAP uses r11 retrieval §§3.4/6 and r14 backbones-sweep/bs_c13. Older C13 code was read, never imported. Shared independent r29rd_c1_core and r29rd_sc2 are allowed earlier re-derivations.',
    'Plain-side mAP uses r17 SC7 untold: unit-normalise three heads in emotion/style/genre order, concatenate and divide by float32 sqrt(3). Q1-side mAP uses the told aspect head as r14 SC6. Two arm-level picks are repeated per aspect; both use told-head R0 validation ranges.',
    'AMI/SC9: float16 exports promoted to float32 without another normalization, MiniBatchKMeans K=9/27/10, n_init=3, batch_size=4096, random_state=0/1/2, eight threads. Cluster all painting rows then mask negative labels. Mean over states then seeds. AMI range is max minus min across five validation seeds.',
    'mAP read-out menu: none, center, PCA-whiten64/256/1024 (capped at width), crossed with k=0/1/3/5/10. Covariance float64, divisor n-1, descending eigenvectors, epsilon=1e-6*largest eigenvalue, float64 W and normalization then one float32 cast; similarities float64 from float32 rows (r11 rt_readout/rt_metrics docstrings). Fit on every fit row without labels. Parity halves precede masking and their means receive equal weight. Final map fits all validation rows.',
    'mAP pick: mean gain over three aspects in told R0 range units, guard against losing more than one range; 0.001 score tie band, R0 first, no expansion, none/center/whiten, smaller k, larger dimension. Permitted independent C13 comparison epsilon 1e-12. Zero R0 range invalidates selection. Query/gallery have labelled rows only; own painting removed; exact ties preserve sorted painting order; AP float64.',
    'Bootstrap: fresh default_rng(0) per comparison. AMI 2000 draws: all painting rows then five candidate seeds; three fixed partitions per seed, re-score repeated rows, mask missing labels after resampling. mAP 10000 draws: labelled query rows then five seeds; gallery fixed. Unseeded plain/Q1 comparators draw no seeds. numpy linear percentiles at 2.5/97.5.',
    'SC9 seeds_ahead is the list of matched seed IDs s for which leader group[s] > runner_up group[s] strictly; tied seeds do not count. Highest exact mean wins; an exact tie involving siglip1 goes to siglip1; otherwise stable siglip2 then clip_l14 order resolves the unspecified tie. close_call is strict margin < leader five-seed range.',
    'K* SC9 groups are read from sc2_rederived.json, never recomputed. Qwen3 c* uses r28 recorded score as information only. Frozen targets are accessed only after deriving corresponding values; targets never choose a result.',
    'Image forward is the img.trunk Linear/GELU then img.output Linear and L2 evaluation path of allowed r29_model.py, applied independently from checkpoint tensors. Sidecar widths/store identity checked. r29_model.py is read, not imported, because its imports include forbidden r29/r28 modules. Float16 gate: >=99.5% equal, all other elements adjacent float16 values.',
    'Frozen --frozen-units replay locates old independent C13 per-query AP caches by their input/source identity, never by point values. It separately recomputes a bounded query sample; full bootstrap can be checked with frozen APs, but this cannot certify full scoring. No stage-2 held data in val or replay. held requires controller --picks-committed and complete validation picks; this asserts an external commitment, makes no commit. Incomplete replays do not certify held cells. Missing/failed gates or picks give INCONCLUSIVE cells, counted as neither WIN nor LEVEL; report inconclusive_above per D22.',
    'Subset --query-limit keeps full galleries and selects evenly spaced query positions; --quick uses only none+k0, center+k0, center+k10 and cannot write final picks or certify picks. Full runs have no subset options. GPU affects only tower/ranking; AMI and bootstrap remain CPU.',
]


def sha(path):
    return rd.sha256(Path(path))


def track(path):
    path = Path(path).resolve()
    if str(path) not in INPUTS:
        INPUTS[str(path)] = sha(path)
    return path


def read_json(path):
    return json.loads(track(path).read_text())


def source_id():
    return hashlib.sha256(''.join(sha(p) for p in (__file__, C.__file__, rd.__file__, S.__file__)).encode()).hexdigest()


def output_path(path):
    p = Path(path).resolve()
    if p.parent != HERE or not (p.name.startswith('r29rd_s2e') or p.name in ('s2e_val.json', 's2e_held.json')):
        raise ValueError('Output must stay in this job\'s rederive files')
    return p


def save(path, doc):
    doc = {**doc, 'notes': NOTES, 'inputs': dict(sorted(INPUTS.items())),
           'source_sha256': source_id(), 'device': C.DEVICE,
           'versions': dict(numpy=np.__version__, torch=torch.__version__, sklearn=sklearn.__version__)}
    rd.write_json(output_path(path), doc)
    return doc


def array_id(x):
    x = np.ascontiguousarray(x)
    return str((x.shape, x.dtype.str)) + hashlib.sha256(x.tobytes()).hexdigest()


def cache(tag, xs, fn):
    identity = [array_id(x) if isinstance(x, np.ndarray) else x for x in xs]
    key = hashlib.sha256(json.dumps([source_id(), C.DEVICE, identity], sort_keys=True).encode()).hexdigest()
    folder = HERE / 'r29rd_s2e_cache'
    folder.mkdir(exist_ok=True)
    p = folder / f'{tag}__{key}.npz'
    if p.exists():
        track(p)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    started = time.monotonic()
    result = fn()
    np.savez(p, **result)
    track(p)
    print(f'{tag}: {time.monotonic()-started:.2f}s', flush=True)
    return result


def names_hash(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


@lru_cache(maxsize=2)
def frame(split):
    """Original extraction-ID painting split, labels independently joined by name."""
    ann = read_json(rd.ANNOTATIONS)
    meta = read_json(rd.FEATURES / 'metadata.json')
    files = [track(rd.FEATURES / 'shards' / f'shard_{s:05d}' / 'sample_ids.npy') for s in range(meta['num_shards'])]
    ids = np.concatenate([np.load(p, allow_pickle=False) for p in files])
    assert len(np.unique(ids)) == len(ids) == len(ann) == 308723
    groups, indices = {}, []
    for sid in ids:
        indices.append(groups.setdefault(ann[int(sid)]['painting'], len(groups)))
    indices = np.asarray(indices)
    counts = np.bincount(indices)
    order = np.random.default_rng(42).permutation(len(counts))
    shares = counts[order] / counts.sum()
    starts = np.cumsum(shares)-shares
    parts = np.empty(len(counts), dtype=np.int64)
    parts[order] = np.searchsorted(np.cumsum([.7, .1, .2])[:2], starts, side='right')
    wanted = np.sort(ids[parts[indices] == (1 if split == 'val' else 2)])
    names = sorted({ann[int(i)]['painting'] for i in wanted})
    expected = (6152, 30872, rd.VAL_SHA256, 4960) if split == 'val' else (12281, 61744, HELD_SHA, 9949)
    assert (len(names), len(wanted), names_hash(names)) == expected[:3]
    lookup = {n: i for i, n in enumerate(names)}
    tp = np.asarray([lookup[ann[int(i)]['painting']] for i in wanted])
    ec = {v: i for i, v in enumerate(sorted({r['emotion'] for r in ann}))}
    sc = {v: i for i, v in enumerate(sorted({r['art_style'] for r in ann}))}
    votes = np.zeros((len(names), len(ec)), dtype=np.int64)
    np.add.at(votes, (tp, [ec[ann[int(i)]['emotion']] for i in wanted]), 1)
    styles = {}
    for row in ann:
        assert styles.setdefault(row['painting'], row['art_style']) == row['art_style']
    genres = {}
    for file in ('genre_train.csv', 'genre_val.csv'):
        with track(rd.GENRE_DIR / file).open() as f:
            for relative, raw in csv.reader(f):
                name, label = Path(relative).stem, int(raw)
                if genres.setdefault(name, label) != label:
                    genres[name] = -1
    ys = dict(emotion=votes.argmax(1), style=np.asarray([sc[styles[n]] for n in names]),
              genre=np.asarray([genres.get(n, -1) for n in names]))
    assert int((ys['genre'] >= 0).sum()) == expected[3]
    return dict(names=names, ids=wanted, tp=tp, labels=ys)


def export_path(encoder, split, aspect, seed, frozen=False):
    if frozen:
        root = R14 / 'cstar' / encoder
        return root / (f'train/k11=40/R6-5/{aspect}__{S.GRAPHS[aspect]}__s{seed}.npz' if split == 'val'
                       else f'held/k11=40/{aspect}__s{seed}.npz')
    if split == 'held' and not ALLOW_HELD:
        raise RuntimeError('Stage-2 held is controller-only after committed validation picks')
    return R29 / encoder / split / ARMS[encoder] / f'{aspect}__s{seed}.npz'


def export(encoder, split, aspect, seed, frozen=False):
    p = export_path(encoder, split, aspect, seed, frozen)
    info = read_json(p.with_suffix('.json'))
    fr = frame(split)
    digests = []
    for doc in (info, info.get('config', {}), info.get('config', {}).get('graph_identity', {})):
        for key in ('list_sha256', 'held_sha256', 'names_sha256', 'painting_names_sha256'):
            val = doc.get(key)
            if isinstance(val, dict):
                val = val.get('dev', val.get('val')) if split == 'val' else val.get('held')
            if isinstance(val, str):
                digests.append(val)
    assert digests and all(d == names_hash(fr['names']) for d in digests), (p, 'row order mismatch', digests)
    for doc in (info, info.get('config', {})):
        assert doc.get('aspect', aspect) == aspect and doc.get('seed', seed) == seed
    with np.load(track(p), allow_pickle=False) as z:
        x = z['img' if 'img' in z else 'emb']
    assert x.dtype == np.float16 and x.shape == (len(fr['names']), 2048) and np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1)-1)) < 1e-3
    if isinstance(info.get('output_sha256'), str):
        assert info['output_sha256'] == sha(p), (p, 'output_sha256')
    # r14 held sidecar sha256 denotes the img ARRAY, not the archive.
    if frozen and split == 'held' and isinstance(info.get('sha256'), str):
        assert info['sha256'] == hashlib.sha256(x.tobytes()).hexdigest()
    expected = info.get('arrays', {}).get('img', {}).get('sha256')
    if expected:
        assert expected == hashlib.sha256(x.tobytes()).hexdigest()
    return x


def stored(folder, split):
    folder, fr = Path(folder), frame(split)
    if (folder / 'index.json').is_file():
        idx = read_json(folder / 'index.json')
        keys = [r['painting'] for r in idx['images']]
    else:
        with np.load(track(folder / 'ids.npz'), allow_pickle=False) as z:
            keys = z['painting_names'].tolist()
    positions = rd.join_rows(keys, fr['names'])
    x = np.load(track(folder / 'img.npy'), mmap_mode='r', allow_pickle=False)
    return C.unit(x[positions])


def plain(encoder, split):
    return stored(rd.STORE_ROOT / STORES[encoder], split)


def teacher(aspect, split):
    suffix = 's2img' if split == 'val' else 'held'
    return stored(rd.STORE_ROOT / 'qwen3vl_emb_2b_aspect_prompts' / f'{INST[aspect]}_{suffix}', split)


def untold(heads):
    return (np.concatenate([C.unit(heads[a]) for a in ASPECTS], axis=1) / np.float32(np.sqrt(3))).astype(np.float32)


def parts(x, aspect):
    return cache('parts_'+aspect, [x, S.CLASS_COUNTS[aspect]],
                 lambda: dict(parts=C.partitions(x, S.CLASS_COUNTS[aspect])))['parts']


def ami_value(p, y):
    good = y >= 0
    return float(np.mean([adjusted_mutual_info_score(y[good], state[good], average_method='arithmetic') for state in p]))


def unit64(x):
    z=np.asarray(x,dtype=np.float64)
    return z / np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)


def map_fit(x, w):
    if w=='none':return None,None
    z=np.asarray(x,dtype=np.float64)
    mu=z.mean(0)
    if w=='center':return mu,None
    from scipy.linalg import eigh
    cov=(z-mu).T@(z-mu)/(len(z)-1)
    size=min(int(w.removeprefix('whiten')),z.shape[1])
    values,vectors=eigh(cov,subset_by_index=(z.shape[1]-size,z.shape[1]-1),driver='evr')
    values,vectors=values[::-1],vectors[:,::-1]
    eps=1e-6*max(float(values[0]),1e-12)
    return mu,vectors / np.sqrt(np.maximum(values,0)+eps)


def map_apply(x, fitted):
    mu,w=fitted
    z=np.asarray(x,dtype=np.float64)
    if mu is not None:z=z-mu
    if w is not None:z=z@w
    return unit64(z).astype(np.float32)


def ranked_maps(x, y, ks, limit=0, batch_size=128):
    """r11 docstrings: float64 similarities from float32 rows, stable ties."""
    valid=np.flatnonzero(y>=0)
    y=y[valid]
    x=np.asarray(x[valid],dtype=np.float64)
    queries=rd.selected_queries(len(x),limit)
    out={k:[] for k in ks}
    gallery=torch.as_tensor(x,device=C.DEVICE) if C.DEVICE=='cuda' else None
    def product(q):
        if C.DEVICE=='cuda':
            return (torch.as_tensor(q,device=C.DEVICE)@gallery.T).cpu().numpy()
        return q@x.T
    def rank(scores, chosen):
        scores[np.arange(len(chosen)),chosen]=-np.inf
        order=np.argsort(-scores,axis=1,kind='stable')
        return order[order!=chosen[:,None]].reshape(len(chosen),len(x)-1)
    for start in range(0,len(queries),batch_size):
        chosen=queries[start:start+batch_size]
        q=x[chosen]
        order=rank(product(q),chosen)
        for k in ks:
            final=order
            if k:
                expanded=unit64(q+x[order[:,:k]].sum(1))
                final=rank(product(expanded),chosen)
            out[k].append(rd.ranked_ap(final,y[chosen],y))
    return {k:np.concatenate(v) for k,v in out.items()}


def image_ap(v, h, y, pick, args, tag='ap'):
    # Callers provide the protocol's representation: told unit(raw), untold join,
    # plain unit(store). Never silently normalize those float32 inputs again.
    def calc():
        w,k=C.parse_readout(pick)
        mapped=map_apply(h,map_fit(v,w))
        return dict(ap=ranked_maps(mapped,y,(k,),args.query_limit,args.batch_size)[k])
    return cache(tag,[v,h,y,pick,args.query_limit,args.batch_size],calc)['ap']


def raw_ap(x,y,args):
    return cache('val_r0',[x,y,args.query_limit,args.batch_size],
        lambda:dict(ap=ranked_maps(C.unit(x),y,(0,),args.query_limit,args.batch_size)[0]))['ap']


def crossfit(x, labels, menu, args):
    """Input is already the appropriate told or untold representation."""
    halves={a:{r:[] for r in menu} for a in labels}
    needed=list(dict.fromkeys(C.parse_readout(r)[0] for r in menu))
    for parity in (0,1):
        mask=np.arange(len(x))%2==parity
        fit,test=x[~mask],x[mask]
        spectral=None
        for w in needed:
            if w.startswith('whiten'):
                if spectral is None:
                    maximum=max(int(z.removeprefix('whiten')) for z in needed if z.startswith('whiten'))
                    spectral=map_fit(fit,f'whiten{maximum}')
                fitted=spectral[0],spectral[1][:,:int(w.removeprefix('whiten'))]
            else:fitted=map_fit(fit,w)
            mapped=map_apply(test,fitted)
            ks=tuple(C.parse_readout(r)[1] for r in menu if C.parse_readout(r)[0]==w)
            for a,y in labels.items():
                aps=ranked_maps(mapped,y[mask],ks,args.query_limit,args.batch_size)
                for k,ap in aps.items():halves[a][f'{w}+k{k}'].append(float(np.nanmean(ap)))
    return {a:{r:float(np.mean(v)) for r,v in rows.items()} for a,rows in halves.items()}


def forward(encoder, aspect, x):
    """Evaluate our independent image tower, per allowed r29_model.py contract."""
    p = R29 / encoder / 'weights' / ARMS[encoder] / f'{aspect}__s0.pt'
    info = read_json(p.with_suffix('.json'))
    cfg, rb = info['config'], info['config']['rb']
    assert cfg['aspect'] == aspect and cfg['seed'] == 0
    width = 1152 if encoder == 'siglip2' else 768
    assert rb['width'] == rb['cap_width'] == width and rb['cap_source'] == ARMS[encoder]
    idx = track(rd.STORE_ROOT / (STORES[encoder]+'_capprefix') / 'index.json')
    assert rb['cap_store_sha256'] == sha(idx)
    assert cfg['names_sha256']['dev'] == names_hash(frame('val')['names'])
    state = torch.load(track(p), map_location='cpu', weights_only=True)
    for side in ('img', 'cap'):
        for key, shape in [('trunk.0.weight', (1024, width)), ('trunk.0.bias', (1024,)),
                           ('output.weight', (2048, 1024)), ('output.bias', (2048,))]:
            z = state[f'{side}.{key}']
            assert z.shape == shape and z.dtype == torch.float32 and torch.isfinite(z).all()
    actual = C.tower(plain(encoder, 'val'), state, 'img')
    result = S.forward_comparison(actual, x)
    q = actual.astype(np.float16)
    upper = np.nextafter(x, np.float16(np.inf)).astype(np.float32)
    lower = np.nextafter(x, np.float16(-np.inf)).astype(np.float32)
    result['all_within_one_float16_step'] = bool(np.all((q.astype(np.float32) >= lower) & (q.astype(np.float32) <= upper)))
    result['pass'] = result['share_equal'] >= .995 and result['all_within_one_float16_step']
    return result


def sc9(groups, kstar, qwen):
    all_groups = dict(siglip1=kstar, **groups)
    assert all(np.shape(g) == (5,) and np.isfinite(g).all() for g in all_groups.values())
    means = {e: float(np.mean(g)) for e, g in all_groups.items()}
    ranked = sorted(all_groups, key=lambda e: (-means[e], ('siglip1','siglip2','clip_l14').index(e)))
    leader, runner = ranked[:2]
    margin = means[leader]-means[runner]
    spread = float(np.ptp(all_groups[leader]))
    ahead = [s for s in SEEDS if all_groups[leader][s] > all_groups[runner][s]]
    return dict(kstar_per_seed=list(kstar), means=means, pick=leader, runner_up=runner, margin=margin,
                seeds_ahead=ahead, n_seeds_ahead=len(ahead), leader_range=spread, close_call=margin < spread,
                qwen_info=qwen)


def comparison(replays, file, key, actual, expected, family, tol):
    if isinstance(actual, str) or isinstance(expected, str):
        diff = 0. if actual == expected else None
        passed = actual == expected
    else:
        diff = abs(float(actual)-float(expected))
        passed = diff <= tol
    row = dict(target_record=str(file), key=key, value=actual, recorded_value=expected,
               absolute_difference=diff, family=family, tolerance=tol, passed=passed)
    replays.append(row)
    return row


def validation_encoder(encoder, args, frozen=False, picks=True):
    fr = frame('val')
    cells = {f'{m}/{a}': [] for a in args.aspects for m in ('ami','map')}
    tables = {side: {a: [] for a in ASPECTS} for side in ('plain','q1')}
    forward_checks = {}
    for seed in args.seeds:
        heads = {}
        for aspect in (ASPECTS if picks else args.aspects):
            x = export(encoder, 'val', aspect, seed, frozen)
            heads[aspect] = x
            y = fr['labels'][aspect]
            if aspect in args.aspects:
                av = ami_value(parts(x, aspect), y)
                ap = raw_ap(x,y,args)
                cells['ami/'+aspect].append(av)
                cells['map/'+aspect].append(float(np.mean(ap)))
            if not frozen and seed == 0:
                forward_checks[aspect] = forward(encoder, aspect, x)
            if picks:
                menu = ['none+k0','center+k0','center+k10'] if args.quick else C.menu(x.shape[1])
                result = cache('crossfit_told', [x,y,menu,args.query_limit,args.batch_size],
                    lambda: {r: np.asarray(v) for r,v in crossfit(C.unit(x),{aspect:y},menu,args)[aspect].items()})
                tables['q1'][aspect].append({r: float(v) for r,v in result.items()})
            print(f'{encoder} val {aspect}/s{seed} done', flush=True)
        if picks:
            joined = untold(heads)
            menu = ['none+k0','center+k0','center+k10'] if args.quick else C.menu(joined.shape[1])
            ys = fr['labels']
            def calc():
                res = crossfit(joined,ys,menu,args)
                return {a+'/'+r: np.asarray(v) for a, table in res.items() for r,v in table.items()}
            result = cache('crossfit_untold', [joined,*[ys[a] for a in ASPECTS],menu,args.query_limit,args.batch_size],calc)
            for a in ASPECTS:
                tables['plain'][a].append({r.split('/',1)[1]:float(v) for r,v in result.items() if r.startswith(a+'/')})
    per_seed = np.mean(list(cells.values()), axis=0, dtype=np.float64).tolist()
    result = dict(groups=dict(per_seed=per_seed, cells=cells, mean=float(np.mean(per_seed))))
    if picks:
        ranges = {a:float(np.ptp(cells['map/'+a])) for a in ASPECTS}
        selected, diagnostics = {}, {}
        for side in ('plain','q1'):
            readouts = list(tables[side]['emotion'][0])
            rows = {r:{a:float(np.mean([t[r] for t in tables[side][a]])) for a in ASPECTS} for r in readouts}
            picked, scores, kept = C.select_image(rows, ranges) if not args.query_limit and len(args.seeds)==5 else (None,{}, {})
            selected[side] = picked
            diagnostics[side] = dict(table=rows, scores=scores, kept=kept, pick=picked)
        result['picks'] = dict(image_map={a:dict(pick_plain_side=selected['plain'],pick_q1_side=selected['q1'],r=ranges[a]) for a in ASPECTS},
                               ami_range={a:float(np.ptp(cells['ami/'+a])) for a in ASPECTS})
        result['image_tables'] = diagnostics
    return result, forward_checks


def validation_gate(reports, encoder):
    rows=[r for d in reports for r in d.get('replays',[]) if encoder in r['key'] or encoder in Path(r['target_record']).stem]
    families=('sc9_ami','sc9_map','sc9_group','ami_range','map_range','pick_plain','pick_q1')
    return all(any(r['family']==f for r in rows) and all(r['passed'] for r in rows if r['family']==f) for f in families)


def check_inputs(args):
    missing=[]
    for e in args.encoders:
        for a in ASPECTS:
            for seed in SEEDS:
                for p in (export_path(e,'val',a,seed),R29/e/'weights'/ARMS[e]/f'{a}__s{seed}.pt'):
                    for q in (p,p.with_suffix('.json')):
                        if not q.is_file():missing.append(str(q))
        for p in (rd.STORE_ROOT/(STORES[e]+'_capprefix')/'index.json',rd.STORE_ROOT/STORES[e]/'img.npy'):
            if not p.is_file():missing.append(str(p))
    return dict(kind='input_availability',complete=False,ready=not missing,missing_inputs=missing,replays=[])


def do_val(args):
    encoders, checks = {}, {}
    reports=[read_json(p) for p in args.replay_reports if p.exists()]
    assert all(d['source_sha256']==source_id() for d in reports), 'Stale replay source'
    for e in args.encoders:
        encoders[e],checks[e] = validation_encoder(e,args)
    gates={e:validation_gate(reports,e) for e in args.encoders}
    complete = full_selection(args) and all(c['pass'] for row in checks.values() for c in row.values()) and all(gates.values())
    if not full_selection(args) and args.output.name == 's2e_val.json':
        raise ValueError('Incomplete diagnostics cannot write s2e_val.json')
    result = dict(kind='validation',complete=complete,encoders=encoders,forward_check=checks,replays=[r for d in reports for r in d['replays']],replay_gates=gates)
    if full_selection(args):
        kstar = read_json(HERE/'sc2_rederived.json')
        assert kstar['kstar']=='siglip1_prefix' and len(kstar['per_seed']['image'])==5
        qwen = read_json(VALPICK)['encoders']['qwen']['score']
        result['sc9'] = sc9({e:encoders[e]['groups']['per_seed'] for e in ENCODERS},kstar['per_seed']['image'],qwen)
    return result


def full_selection(args):
    return args.encoders==list(ENCODERS) and args.aspects==list(ASPECTS) and args.seeds==list(SEEDS) and not args.query_limit and not args.quick and args.sides==['plain','q1']


def replay_validation(args, records, picks):
    results = {}
    for e in args.encoders:
        result,_ = validation_encoder(e,args,True,picks)
        results[e] = result
        # Compare only after deriving cells, ranges and picks.
        doc = read_json(VALPICK)['encoders'][e]
        for cell, values in result['groups']['cells'].items():
            for s,value in zip(args.seeds,values):
                if cell.startswith('ami/') or not args.query_limit:
                    comparison(records,VALPICK,f'encoders/{e}/cells/{cell}/per_seed/{s}',value,doc['cells'][cell]['per_seed'][s],'sc9_'+cell.split('/')[0],1e-6)
        if args.aspects==list(ASPECTS) and args.seeds==list(SEEDS) and not args.query_limit:
            comparison(records,VALPICK,f'encoders/{e}/score',result['groups']['mean'],doc['score'],'sc9_group',1e-6)
        if picks:
            old = read_json(REC14/f'stage1_{e}.json')
            oldami = read_json(REC17/'stage1_ranges.json')
            oldpick = read_json(REC17/'stage1_picks.json')
            complete = args.seeds==list(SEEDS) and not args.query_limit and not args.quick
            for a in ASPECTS:
                if complete:
                    comparison(records,REC17/'stage1_ranges.json',f'cstar/artelingo/{e}/{a}/ami/r',result['picks']['ami_range'][a],oldami[f'cstar/artelingo/{e}/{a}/ami']['r'],'ami_range',1e-9)
                    comparison(records,REC14/f'stage1_{e}.json',f'artret_ranges/ranges/{a}',result['picks']['image_map'][a]['r'],old['artret_ranges']['ranges'][a],'map_range',1e-9)
            if complete:
                comparison(records,REC17/'stage1_picks.json',f'untold/artelingo/{e}/pick',result['picks']['image_map']['emotion']['pick_plain_side'],oldpick[f'untold/artelingo/{e}']['pick'],'pick_plain',0.)
                comparison(records,REC14/f'stage1_{e}.json','artret_picks/cstar/pick',result['picks']['image_map']['emotion']['pick_q1_side'],old['artret_picks']['cstar']['pick'],'pick_q1',0.)
    return results


def target(encoder, aspect, metric, side):
    if metric=='ami' or side=='plain':
        file = REC17/'outcomes.json'
        sc = 'SC1' if metric=='ami' and side=='plain' else 'SC3' if metric=='ami' else 'SC7'
        row = next(c for c in read_json(file)['cells'] if c['encoder']==encoder and c['aspect']==aspect and c['set']=='artelingo'
                   and c['metric']==metric and c['sc']==sc and c.get('kind')!='fixed20')
        return file,row['id'],dict(value=row['cand'],comparator_value=row['comp'],d=row['d'],lo=row['lo'],hi=row['hi'],r=row['r']),row
    file = REC14/'verdicts.json'
    row = next(c for c in read_json(file)['cells'] if c['encoder']==encoder and c['aspect']==aspect and c['sc']=='SC6'
               and c['task']=='ret' and c['data']=='artelingo_held')
    return file,f'SC6/{encoder}/{aspect}/ret',dict(value=row['candidate_value'],comparator_value=row['comparator_value'],
               d=row['d'],lo=row['lo'],hi=row['hi'],r=row['r']),row


def frozen_picks(encoder):
    old = read_json(REC14/f'stage1_{encoder}.json')
    untold_pick = read_json(REC17/'stage1_picks.json')[f'untold/artelingo/{encoder}']['pick']
    ranges = read_json(REC17/'stage1_ranges.json')
    return dict(image_map={a:dict(pick_plain_side=untold_pick,pick_q1_side=old['artret_picks']['cstar']['pick'],
                                 r=old['artret_ranges']['ranges'][a]) for a in ASPECTS},
                ami_range={a:ranges[f'cstar/artelingo/{encoder}/{a}/ami']['r'] for a in ASPECTS})


def comparator_ap(encoder, aspect, side, args):
    fr = frame('held')
    y = fr['labels'][aspect]
    if side=='q1':
        p = R11/f'stage5/perquery/instructed__{aspect}__h{aspect}__s-.npz'
        with np.load(track(p),allow_pickle=False) as z:
            assert np.array_equal(z['rows'],np.flatnonzero(y>=0))
            ap = z['ap']
        pick = read_json(R11/'stage4/picks.json')['instructed']
        if args.query_limit:
            ap=ap[rd.selected_queries(len(ap),args.query_limit)]
        return ap,pick
    # r17 plain comparator at its frozen r11/r14 validation pick; rebuild from store.
    pick = read_json(REC14/f'stage1_{encoder}.json')['artret_picks']['frozen']['pick']
    return image_ap(plain(encoder,'val'),plain(encoder,'held'),y,pick,args,'plain_ap'),pick


def frozen_unit_path(encoder, v, h, pick, y):
    # Recreate only the published cache address algorithm of permitted ns_c13.Work.array.
    root=Path('/data/SSD2/cosir_r17/20261103_new_scores/c13')
    meta=read_json(root/'derived'/f'cells:ret:artelingo:{encoder}.json')
    digest=hashlib.sha256(meta['source_sha256'].encode())
    for x in (frame('val')['names'],frame('held')['names'],v,h,pick,y):
        if isinstance(x,np.ndarray):
            digest.update(str((x.shape,x.dtype.str)).encode())
            digest.update(np.ascontiguousarray(x).tobytes())
        else:digest.update(json.dumps(x,sort_keys=True).encode())
    p=root/'cache'/f'picked_ap__{digest.hexdigest()}.npz'
    with np.load(track(p),allow_pickle=False) as z:return z['value']


def frozen_ap(encoder, aspect, side, seed):
    picks=frozen_picks(encoder)
    if side=='plain':
        vectors=[untold({a:export(encoder,split,a,seed,True) for a in ASPECTS}) for split in ('val','held')]
    else:
        vectors=[C.unit(export(encoder,split,aspect,seed,True)) for split in ('val','held')]
    return frozen_unit_path(encoder,*vectors,picks['image_map'][aspect]['pick_'+side+'_side'],frame('held')['labels'][aspect])


def frozen_comparator_ap(encoder,aspect,side):
    if side=='q1':
        p=R11/f'stage5/perquery/instructed__{aspect}__h{aspect}__s-.npz'
        with np.load(track(p),allow_pickle=False) as z:return z['ap']
    pick=read_json(REC14/f'stage1_{encoder}.json')['artret_picks']['frozen']['pick']
    return frozen_unit_path(encoder,plain(encoder,'val'),plain(encoder,'held'),pick,frame('held')['labels'][aspect])


def test_encoder(encoder, args, picks, frozen=False):
    fr = frame('held')
    candidates = {a:dict(ami=[],plain=[],q1=[]) for a in args.aspects}
    metrics = ('ami','map') if args.family=='all' else (args.family,)
    for seed in SEEDS:
        heads_h,heads_v = {},{}
        for a in (ASPECTS if 'map' in metrics else args.aspects):
            h = export(encoder,'held',a,seed,frozen)
            heads_h[a]=h
            y = fr['labels'][a]
            if a in args.aspects and 'ami' in metrics:
                candidates[a]['ami'].append(parts(h,a))
            if 'map' in metrics:
                v = export(encoder,'val',a,seed,frozen)
                heads_v[a]=v
                if a in args.aspects and 'q1' in args.sides:
                    pick=picks['image_map'][a]['pick_q1_side']
                    candidates[a]['q1'].append(image_ap(C.unit(v),C.unit(h),y,pick,args,'told_ap'))
        if 'map' in metrics and 'plain' in args.sides:
            v,h=untold(heads_v),untold(heads_h)
            for a in args.aspects:
                pick=picks['image_map'][a]['pick_plain_side']
                candidates[a]['plain'].append(image_ap(v,h,fr['labels'][a],pick,args,'untold_ap'))
        print(f'{encoder} {"frozen" if frozen else "candidate"} held s{seed} complete',flush=True)
    cells=[]
    for a in args.aspects:
        y=fr['labels'][a]
        for metric in metrics:
            for side in args.sides:
                started=time.monotonic()
                if metric=='ami':
                    other=parts(plain(encoder,'held') if side=='plain' else teacher(a,'held'),a)
                    stats=C.ami_interval(np.stack(candidates[a]['ami']),other,y,args.ami_draws,args.workers)
                    r=picks['ami_range'][a]
                    n=len(y)
                    extras=dict(n_labelled=int((y>=0).sum()),n_draws=args.ami_draws)
                else:
                    other,other_pick=comparator_ap(encoder,a,side,args)
                    stats=C.retrieval_interval(np.stack(candidates[a][side]),other,draws=args.ret_draws)
                    r=picks['image_map'][a]['r']
                    n=len(other)
                    extras=dict(candidate_readout=picks['image_map'][a]['pick_'+side+'_side'],comparator_readout=other_pick,n_draws=args.ret_draws)
                scoring_check=None
                if frozen and metric=='map' and args.frozen_units:
                    own_c=np.stack(candidates[a][side])
                    saved_c=np.stack([frozen_ap(encoder,a,side,s) for s in SEEDS])
                    saved_b=frozen_comparator_ap(encoder,a,side)
                    qi=rd.selected_queries(len(saved_b),args.query_limit)
                    scoring_check=dict(candidate_sample_max_abs_diff=float(np.max(np.abs(own_c-saved_c[:,qi]))),
                                       comparator_sample_max_abs_diff=float(np.max(np.abs(other-saved_b[qi]))),
                                       n_queries=len(qi),full_scoring=not args.query_limit)
                    stats=C.retrieval_interval(saved_c,saved_b,draws=args.ret_draws)
                    n=len(saved_b)
                row=dict(id=f'{encoder}/{metric}/{a}/{side}',encoder=encoder,metric=metric,aspect=a,side=side,
                         **stats,r=r,outcome=C.outcome(stats['lo'],r),n=n,**extras)
                if scoring_check is not None:row['scoring_check']=scoring_check
                cells.append(row)
                print(f'{row["id"]} interval {time.monotonic()-started:.1f}s',flush=True)
    return cells


def do_replay(args):
    records,encoders,cells=[],{},[]
    if args.family in ('groups','val','all'):
        encoders=replay_validation(args,records,args.family!='groups')
    if args.family in ('ami','map','all'):
        for e in args.encoders:
            picks=frozen_picks(e)
            rows=test_encoder(e,args,picks,True)
            for row in rows:
                p,key,values,old=target(e,row['aspect'],row['metric'],row['side'])
                full_interval=(args.ami_draws==2000 if row['metric']=='ami' else args.ret_draws==10000 and (not args.query_limit or args.frozen_units))
                full_point=row['metric']=='ami' or not args.query_limit or args.frozen_units
                keys=['value','comparator_value','d','r'] if full_point else []
                if full_interval:keys+=['lo','hi']
                for k in keys:
                    comparison(records,p,key+'/'+k,row[k],values[k],row['metric']+'_'+('interval' if k in ('lo','hi') else 'point'),1e-9)
                row['full_interval']=full_interval
                row['full_scoring']=row['metric']=='ami' or not args.query_limit
            cells.extend(rows)
    summary={}
    for family in sorted({r['family'] for r in records}):
        group=[r for r in records if r['family']==family]
        summary[family]=dict(n=len(group),max_abs_diff=max((r['absolute_difference'] for r in group if r['absolute_difference'] is not None),default=None),passed=all(r['passed'] for r in group))
    return dict(kind='frozen_replay',complete=full_selection(args) and args.family=='all' and args.ami_draws==2000 and args.ret_draws==10000,
                encoders=encoders,cells=cells,replays=records,summary=summary,seeds=args.seeds,aspects=args.aspects,
                query_limit=args.query_limit,quick=args.quick,ami_draws=args.ami_draws,ret_draws=args.ret_draws)


def ladder(cells):
    plain=[c for c in cells if c['side']=='plain']
    q1=[c for c in cells if c['side']=='q1']
    wp=sum(c['outcome']=='WIN' for c in plain)
    wq=sum(c['outcome'] in ('WIN','LEVEL') for c in q1)
    ip=sum(c['outcome']=='INCONCLUSIVE' for c in plain)
    iq=sum(c['outcome']=='INCONCLUSIVE' for c in q1)
    def level(p,q):return 3 if p==6 and q>=5 else 2 if p==6 else 1 if p>=5 else 0
    l=level(wp,wq)
    return dict(wins_plain=wp,wins_or_level_q1=wq,level=l,inconclusive=ip+iq,
                inconclusive_above=level(wp+ip,wq+iq)>l)


def certified(reports, encoder, aspect, metric, side):
    relevant=[]
    for d in reports:
        for cell in d.get('cells',[]):
            if cell['id']==f'{encoder}/{metric}/{aspect}/{side}' and cell.get('full_interval') and cell.get('full_scoring'):
                relevant.append(d)
    # Replay must contain direct numerical point and interval comparisons for this cell.
    for d in relevant:
        _,key,_,_=target(encoder,aspect,metric,side)
        rows=[r for r in d['replays'] if r['key'].startswith(key+'/')]
        if {r['key'].rsplit('/',1)[1] for r in rows} >= {'value','comparator_value','d','lo','hi','r'} and all(r['passed'] for r in rows):
            return True
    return False


def do_held(args):
    global ALLOW_HELD
    if not args.picks_committed:
        raise ValueError('held requires controller --picks-committed; no stage-2 held file was read')
    doc=read_json(args.picks)
    assert doc['complete'] and doc['source_sha256']==source_id(), 'Incomplete/stale validation picks'
    assert full_selection(args) and args.family=='all' and args.ami_draws==2000 and args.ret_draws==10000
    reports=[read_json(p) for p in args.replay_reports]
    assert all(d['source_sha256']==source_id() for d in reports), 'Stale replay code'
    # Pin all validation input bytes before reading held exports, including dependency source/data files.
    for file,digest in doc['inputs'].items():
        assert sha(track(file))==digest, ('Validation input changed',file)
    ALLOW_HELD=True
    encoders={}
    for e in ENCODERS:
        cells=test_encoder(e,args,doc['encoders'][e]['picks'])
        for row in cells:
            if not certified(reports,e,row['aspect'],row['metric'],row['side']):
                row['outcome']='INCONCLUSIVE'
                row['reason']='Missing or failed complete 1e-9 frozen replay for this cell'
        encoders[e]=dict(cells=cells,sc4=ladder(cells))
    return dict(kind='held',encoders=encoders,replays=[r for d in reports for r in d['replays']],
                picks_sha256=sha(args.picks),complete=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for mode in ('val','held','replay'):
        p=sub.add_parser(mode)
        p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        p.add_argument('--output',type=Path,default=HERE/('s2e_'+mode+'.json' if mode!='replay' else 'r29rd_s2e_replay.json'))
        p.add_argument('--encoders',choices=ENCODERS,nargs='+',default=list(ENCODERS))
        p.add_argument('--aspects',choices=ASPECTS,nargs='+',default=list(ASPECTS))
        p.add_argument('--seeds',type=int,choices=SEEDS,nargs='+',default=list(SEEDS))
        p.add_argument('--query-limit',type=int,default=0)
        p.add_argument('--batch-size',type=int,default=128)
        p.add_argument('--quick',action='store_true')
        p.add_argument('--family',choices=('all','groups','val','ami','map'),default='all')
        p.add_argument('--ami-draws',type=int,default=2000)
        p.add_argument('--ret-draws',type=int,default=10000)
        p.add_argument('--workers',type=int,choices=range(1,9),default=8)
        p.add_argument('--sides',choices=('plain','q1'),nargs='+',default=['plain','q1'])
        p.add_argument('--check-inputs',action='store_true')
        p.add_argument('--frozen-units',action='store_true')
        p.add_argument('--replay-reports',type=Path,nargs='+',default=[HERE/'r29rd_s2e_replay.json'])
        if mode=='held':
            p.add_argument('--picks',type=Path,default=HERE/'s2e_val.json')
            p.add_argument('--picks-committed',action='store_true')
    args=parser.parse_args()
    args.output=output_path(args.output)
    if args.query_limit<0 or args.batch_size<1 or args.ami_draws<1 or args.ret_draws<1:
        parser.error('Invalid numeric option')
    if args.frozen_units and (args.command!='replay' or args.family!='map'):
        parser.error('--frozen-units is only for replay --family map')
    if args.command=='held' and args.check_inputs:parser.error('No stage-2 held check before the controller run')
    if args.family=='val' and args.aspects!=list(ASPECTS):parser.error('read-out picking requires all aspects')
    if args.command=='val' and args.aspects!=list(ASPECTS):parser.error('val picks require all aspects')
    C.set_device(args.device)
    for p in (__file__,C.__file__,rd.__file__,S.__file__):track(p)
    for name in ('2026-11-12-siglip-only-adapter-design.md','2026-10-29-retrieval-design.md',
                 '2026-11-02-backbones-sweep-design.md','2026-11-03-new-scores-design.md'):track(SPEC/name)
    track(HERE.parent/'r29_model.py')
    for name in ('rt_metrics.py','rt_readout.py','rt_stats.py','rt_data.py'):
        track(TEST/'20261029_retrieval'/name)  # Only docstrings inspected; never import.
    start=time.monotonic()
    result=check_inputs(args) if args.check_inputs else {'val':do_val,'held':do_held,'replay':do_replay}[args.command](args)
    result['elapsed_seconds']=time.monotonic()-start
    save(args.output,result)
    print(f'{args.command} complete in {result["elapsed_seconds"]:.1f}s -> {args.output.name}; summary={result.get("summary",{})}',flush=True)


if __name__=='__main__':main()
