"""Independent phase 1. --prepare fits/caches; --score derives every seed-42 pick.

Split the invocations for the four-minute command limit; the complete runs can be
launched by the controller. Each invocation refuses existing outputs.
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
from rd9_inputs import Inputs,RD,OUT,feature_data,save_npz,save_json,check_output,sha_file,verify_reads
from rd9_bundle import split,fit_heads,encoded,prepare_arrays,fit_basis,score_block
from rd9_episodes import PAIRS,FIELDS,concat,build_standard
from rd9_stats import pass_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',default='clip_b32')
    p.add_argument('--pins',type=Path)
    p.add_argument('--stage',choices=['prepare','score'],required=True)
    args=p.parse_args()
    inputs=Inputs(args.backbone,args.pins)
    prefix=OUT/f'rd9_{args.backbone}'
    if args.stage=='prepare':
        data=feature_data(args.backbone)
        sp=split(inputs,data)
        models,post,coefs=fit_heads(inputs,data,sp)
        codes=encoded(inputs,data,sp['selection'])
        bundle=prepare_arrays(data,sp['selection'],post,codes)
        bundle['groups']=sp['groups']
        basis=fit_basis(inputs,data,sp)
        # Build the development value sets and episodes independently of the pinned episodes.
        from src.data.artelingo_splits import artelingo_aspect_labels
        from src.eval.aspect_episodes import eligible_values,episodes_sha256
        from rd9_episodes import pool_rows
        labels=artelingo_aspect_labels(data)
        pool=pool_rows(labels,sp['selection'])
        values={a:eligible_values(labels[a],sp['groups'],pool,30) for a in labels}
        assert [len(values[a]) for a in ('emotion','style','genre')]==[8,23,10]
        episodes=build_standard(labels,sp['groups'],sp['selection'],values,42)
        expected=inputs.json('baselines_seed42',['episodes_sha256'])['episodes_sha256']
        assert {key:episodes_sha256(ep) for key,ep in episodes.items()}==expected
        bundle.update({f'{key}__{field}':getattr(ep,field) for key,ep in episodes.items() for field in FIELDS})
        save_npz(str(prefix)+'_bundle.npz',bundle)
        with check_output(str(prefix)+'_fits.pkl').open('xb') as f: pickle.dump(dict(heads=models,basis=basis),f)
        save_json(str(prefix)+'_prepare.json',dict(inputs=inputs.reads,coef_sha256=coefs,value_sets=values,
                                                 bundle_sha256=sha_file(str(prefix)+'_bundle.npz'),
                                                 fits_sha256=sha_file(str(prefix)+'_fits.pkl')))
        print('phase 1 prepared:',args.backbone)
    else:
        import json
        from src.eval.aspect_episodes import AspectEpisodes,episodes_sha256
        rec=json.loads(Path(str(prefix)+'_prepare.json').read_text())
        verify_reads(rec['inputs'])
        assert sha_file(str(prefix)+'_bundle.npz')==rec['bundle_sha256']
        assert sha_file(str(prefix)+'_fits.pkl')==rec['fits_sha256']
        with np.load(str(prefix)+'_bundle.npz') as z: bundle={k:z[k] for k in z.files}
        with Path(str(prefix)+'_fits.pkl').open('rb') as f: fits=pickle.load(f)
        parts=[AspectEpisodes(a,b,*(bundle[f'{a}__{b}__{field}'] for field in FIELDS)) for a,b,_ in PAIRS]
        ep=concat(parts)
        readers=inputs.readers()
        arrays,picks=score_block(bundle,ep,readers,fits['basis'])
        frozen_taus=inputs.json('rc_tau',['taus'])['taus']
        assert np.allclose(picks['taus'],frozen_taus,rtol=1e-9,atol=1e-15), 'tau disagreement'
        arrays['cl']=bundle['groups'][ep.anchor]
        arrays['pair_index']=np.repeat(np.arange(3),4096)
        arrays['taus']=np.asarray(picks['taus'])
        arrays['aff_fused_cells']=np.asarray(picks['aff'])
        arrays['aff_cf_cells']=np.asarray(picks['cf'])
        arrays['aff_sigma']=np.asarray(picks['sigma_star'])
        stats=pass_record(arrays,sensitivity=True)
        stats['episodes_sha256']={'42':{f'{a}__{b}':episodes_sha256(part) for (a,b,_),part in zip(PAIRS,parts)}}
        stats['seeds']=[42]
        save_npz(str(prefix)+'_seed42_arrays.npz',arrays)
        save_json(str(prefix)+'_phase1.json',dict(backbone=args.backbone,picks=picks,stats=stats,
                                               inputs={**rec['inputs'],**inputs.reads},coef_sha256=rec['coef_sha256'],
                                               value_sets=rec['value_sets']))
        print('phase 1 complete:',args.backbone)


if __name__=='__main__': main()
