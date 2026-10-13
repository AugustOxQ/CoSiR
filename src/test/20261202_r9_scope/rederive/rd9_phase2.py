"""Per-read independent episodes, frozen assembly and pooled P1–P7.

No read-time tuning. Writes independent records in rederive, never an official
verdict or a ledger row. --mode selection replays seed 42 through this same path.
"""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rd9_inputs import Inputs,RD,OUT,HERE,T,feature_data,save_json,save_npz,sha_file,verify_reads,library_sha256
from rd9_bundle import split,encoded,prepare_arrays,score_block
from rd9_scores import unit32
from rd9_episodes import PAIRS,FIELDS,concat,value_codes,build_standard,build_a28,table_arrays
from rd9_stats import pass_record
from src.eval.aspect_episodes import episodes_sha256,AspectEpisodes

READ_BACKBONE={'H6':'qwen3vl_emb_2b','H7':'siglip2_so400m','H8':'clip_b32','H9':'qwen3vl_emb_2b','H10':'siglip2_so400m'}


def a28_data(backbone,table,root=None):
    folder=Path(root or '/data/SSD2/pre_extract/artelingo28')/backbone
    index=json.loads((folder/'index.json').read_text())['texts']
    rows=np.flatnonzero(~table.dropped.to_numpy())
    ids=table.row_id.to_numpy()
    assert np.array_equal(np.array([r['row_id'] for r in index]),ids[rows])
    assert all(str(r['painting'])==str(table.painting.iloc[k]) for k,r in zip(rows,index))
    image=np.load(folder/'img.npy',mmap_mode='r')
    text=np.load(folder/'txt.npy',mmap_mode='r')
    join=np.array([r['img'] for r in index],np.int64)
    assert len(text)==len(index)==len(rows) and len(image)==746
    assert image.ndim==text.ndim==2 and image.shape[1]==text.shape[1]
    assert join.min()>=0 and join.max()<len(image)
    n=len(table)
    img,txt=[np.full((n,image.shape[1]),np.nan,np.float32) for _ in range(2)]
    img[rows]=image[join];txt[rows]=text
    assert np.isfinite(img[rows]).all() and np.isfinite(txt[rows]).all()
    assert np.isnan(img[table.dropped]).all() and np.isnan(txt[table.dropped]).all()
    return SimpleNamespace(img_features=img,txt_features=txt,sample_ids=ids),rows


def hash_guard(hashes,prior):
    current=[digest for pairs in hashes.values() for digest in pairs.values()]
    assert len(current)==len(set(current)), 'repeated episode hash within this read'
    assert not set(current)&set(prior), 'episode hash repeats a prior non-smoke episode'


def h5_parts(archive,per_pair=4096,check_hashes=True):
    names=[f'{a}__{b}' for a,b,_ in PAIRS]
    assert archive['pair_names'].tolist()==names
    idx=archive['pair_index']
    assert np.array_equal(idx,np.repeat(np.arange(3),per_pair))
    for field in FIELDS:
        value=archive[field]
        shape=(3*per_pair,) if field=='anchor' else (3*per_pair,13 if field=='candidates' else 4)
        assert value.shape==shape and value.dtype==np.int64
    result={}
    for j,(a,b,_) in enumerate(PAIRS):
        ep=AspectEpisodes(a,b,*(archive[f][idx==j] for f in FIELDS))
        key=f'{a}__{b}'
        if check_hashes:
            assert str(archive['sha__'+key].item())==episodes_sha256(ep)
        result[key]=ep
    return result


def prior_hashes():
    prior=set()
    for path in (T/'20261030_aspect_baselines/results').glob('baselines_seed*.json'):
        rec=json.loads(path.read_text())
        prior.update(rec['episodes_sha256'].values())
    for s in (52,53,54):
        with np.load(T/f'20261125_artelingo_held_test/results/held_episodes_seed{s}.npz') as z:
            for ep in h5_parts(z).values():
                prior.add(episodes_sha256(ep))
    return prior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--read',choices=list(READ_BACKBONE),required=True)
    p.add_argument('--pins',type=Path)
    p.add_argument('--mode',choices=['read','selection'],default='read')
    p.add_argument('--a28-pins',type=Path,default=HERE/'pins_a28.json')
    args=p.parse_args()
    backbone=READ_BACKBONE[args.read]
    inputs=Inputs(backbone,args.pins)
    prefix=OUT/f'rd9_{backbone}'
    phase1=json.loads(Path(str(prefix)+'_phase1.json').read_text())
    prep=json.loads(Path(str(prefix)+'_prepare.json').read_text())
    assert sha_file(str(prefix)+'_fits.pkl')==prep['fits_sha256']
    with Path(str(prefix)+'_fits.pkl').open('rb') as f: fits=pickle.load(f)
    verify_reads(phase1['inputs'])
    frozen=phase1['picks']
    assert not frozen['switched_off'], 'not run: switched off'
    a28=args.read in ('H8','H9','H10') and args.mode=='read'
    seeds=[42] if args.mode=='selection' else ([55,56,57] if a28 else [52,53,54])
    if a28:
        # All pins are verified without importing the dataset loader of the chain.
        from rd9_inputs import pin_entries,resolve_pin
        entries=pin_entries(json.loads(args.a28_pins.read_text()))
        assert entries
        for rel,digest in entries.items():
            path=resolve_pin(rel)
            assert path.suffix!='.py' and sha_file(path)==digest
        import pandas as pd
        table=pd.read_parquet(HERE/'results/read_set_a28.parquet')
        values=json.loads((HERE/'results/value_sets_a28.json').read_text())
        labels,groups,_=table_arrays(table)
        data,rows=a28_data(backbone,table)
    else:
        data=feature_data(backbone)
        sp=split(inputs,data)
        groups=sp['groups']
        from src.data.artelingo_splits import artelingo_aspect_labels
        labels=artelingo_aspect_labels(data)
        values=phase1['value_sets']
        rows=sp['selection'] if args.mode=='selection' else inputs.npz('held_codes',['held_rows'])['held_rows']
        if args.mode=='read':
            assert len(rows)==61744 and not np.isin(groups[rows],groups[sp['split_train']]).any()
    # The heads and A3 checkpoint are frozen; no fit or pick sees read labels/features.
    post={g:{m:fits['heads'][g][m].predict_proba(unit32(getattr(data,m+'_features')[rows])).astype(np.float32)
             for m in ('img','txt')} for g in fits['heads']}
    codes=encoded(inputs,data,rows)
    bundle=prepare_arrays(data,rows,post,codes)
    readers=inputs.readers()
    pooled,episodes_out,hashes,episode_records={},{},{},{}
    for seed in seeds:
        if a28:
            eps,episode_records[str(seed)]=build_a28(table,values,seed)
        else:
            eps=build_standard(labels,groups,rows,value_codes(values),seed)
        hashes[str(seed)]={key:episodes_sha256(ep) for key,ep in eps.items()}
        if not a28 and args.mode=='read':
            with np.load(T/f'20261125_artelingo_held_test/results/held_episodes_seed{seed}.npz') as original:
                pinned_parts=h5_parts(original)
                for key,ep in eps.items():
                    for field in FIELDS:
                        assert np.array_equal(getattr(ep,field),getattr(pinned_parts[key],field)), 'H5 episode identity'
        ep=concat(list(eps.values()))
        arrays,_=score_block(bundle,ep,readers,fits['basis'],frozen,frozen['taus'])
        arrays['cl']=groups[ep.anchor]
        arrays['pair_index']=np.repeat(np.arange(3),4096)
        arrays['seed_index']=np.full(len(ep.anchor),seeds.index(seed),np.int64)
        for key,v in arrays.items(): pooled.setdefault(key,[]).append(v)
        episodes_out.update({f'seed{seed}__{key}__{field}':getattr(part,field)
                             for key,part in eps.items() for field in FIELDS})
    if a28:
        hash_guard(hashes,prior_hashes())
        if args.read in ('H9','H10'):
            H8=json.loads((OUT/'rd9_H8_read.json').read_text())
            assert hashes==H8['stats']['episodes_sha256'], 'H8 episode hashes differ'
    arrays={key:np.concatenate(parts,axis=1 if '_gate__' in key else 0) for key,parts in pooled.items()}
    assert len(arrays['cl'])==12288*len(seeds)
    stats=pass_record(arrays,a28=a28)
    stats.update(seeds=seeds,episodes_sha256=hashes)
    # Detectable margins from selection variance and the actual pooled cluster sizes.
    _,inv=np.unique(arrays['cl'],return_inverse=True)
    N=len(inv);sum_M2=float(np.sum(np.bincount(inv).astype(np.float64)**2))
    for j,v in phase1['stats']['sensitivity'].items():
        se=float(np.sqrt((v['sigma_a2']*sum_M2+v['sigma_e2']*N)/N**2))
        factor=(3.083 if j in ('P4','P6') else 3.417) if a28 else 3.532
        stats['checks'][j].update(SE=se,x=factor*se,x95=2.80*se)
    suffix=f'rd9_{args.read}_{args.mode}'
    save_npz(OUT/f'{suffix}_arrays.npz',arrays)
    save_npz(OUT/f'{suffix}_episodes.npz',episodes_out)
    save_json(OUT/f'{suffix}.json',dict(library_sha256=library_sha256(),read=args.read,backbone=backbone,mode=args.mode,
                                    phase1_sha256=sha_file(str(prefix)+'_phase1.json'),picks=frozen,stats=stats,
                                    episode_records=episode_records,inputs=inputs.reads))
    print('phase 2 complete:',args.read,args.mode)


if __name__=='__main__': main()
