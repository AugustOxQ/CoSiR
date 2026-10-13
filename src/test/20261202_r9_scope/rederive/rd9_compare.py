"""C13 agreement as R3 §8. Only opens runner outputs after our record exists.

Writes one JSON per read. Missing quantities are differences, never agreements.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from rd9_inputs import RD,T,save_json,sha_file
from rd9_stats import CHECK_IDS


def compare_value(ours,theirs,key):
    if isinstance(ours,bool) or isinstance(theirs,bool):
        return type(ours)==type(theirs) and ours==theirs
    if isinstance(ours,int) or isinstance(theirs,int):
        return ours==theirs
    if key.startswith('picks.') and 'taus' not in key:
        return ours==theirs
    if isinstance(ours,float) and isinstance(theirs,(float,int)):
        if 'taus' in key or 'tau.' in key:
            return bool(np.isclose(ours,theirs,rtol=1e-9,atol=1e-15))
        return bool(abs(ours-theirs)<=1e-9)
    return ours==theirs


def compare_array(ours,theirs):
    return ours.shape==theirs.shape and ours.dtype==theirs.dtype and np.array_equal(ours,theirs)


class Comparison:
    def __init__(self): self.items=[]
    def value(self,key,ours,theirs):
        if isinstance(ours,dict):
            if not isinstance(theirs,dict):
                self.items.append(dict(key=key,equal=False,reason='missing object'));return
            for k,v in ours.items(): self.value(f'{key}.{k}',v,theirs.get(k))
        elif isinstance(ours,(list,tuple)):
            if not isinstance(theirs,(list,tuple)) or len(ours)!=len(theirs):
                self.items.append(dict(key=key,equal=False,reason='sequence length/type'));return
            for j,(a,b) in enumerate(zip(ours,theirs)): self.value(f'{key}.{j}',a,b)
        else:
            equal=theirs is not None and compare_value(ours,theirs,key)
            self.items.append(dict(key=key,equal=bool(equal),ours=ours,theirs=theirs))
    def arrays(self,ours_path,theirs_path,required=None):
        with np.load(ours_path,allow_pickle=False) as ours,np.load(theirs_path,allow_pickle=False) as theirs:
            for key in required or ours.files:
                equal=key in ours.files and key in theirs.files and compare_array(ours[key],theirs[key])
                self.items.append(dict(key='arrays.'+key,equal=bool(equal)))
    def record(self):
        failures=[x for x in self.items if not x['equal']]
        assert self.items
        return dict(agreement=not failures,compared=len(self.items),failed=len(failures),items=self.items)


def compare_stats(comparison,ours,theirs):
    comparison.value('n_episodes',ours['n_episodes'],theirs.get('n_episodes'))
    comparison.value('n_clusters',ours['n_clusters'],theirs.get('n_clusters'))
    comparison.value('seeds',ours['seeds'],theirs.get('seeds'))
    comparison.value('episodes_sha256',ours['episodes_sha256'],theirs.get('episodes_sha256'))
    for j in CHECK_IDS:
        o=ours['checks'][j]
        t=theirs.get('checks',{}).get(j,{})
        for k in ('n','point','ci95','tested'):
            if k=='tested':
                # A seven-check record always tests all seven. A28 records must specify this flag.
                target=t.get('tested',True if 'holm_order' in theirs else None)
            else:
                target=t.get(k,t.get('n_j') if k=='n' else None)
            comparison.value('checks.'+j+'.'+k,o[k],target)
        if o['tested']:
            for k in ('holm_k','ci_holm','passes','own_count_passes','near_boundary','level_two_sided'):
                alias={'passes':'pass','near_boundary':'within_one_of_boundary','level_two_sided':'holm_level_two_sided'}.get(k)
                comparison.value('checks.'+j+'.'+k,o[k],t.get(k,t.get(alias) if alias else None))
    for key in ('holm_order','holm_order_level1','holm_order_level2'):
        if key in ours: comparison.value(key,ours[key],theirs.get(key))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--derived',type=Path,required=True)
    p.add_argument('--reported',type=Path)
    p.add_argument('--derived-arrays',type=Path,required=True)
    p.add_argument('--reported-arrays',type=Path)
    p.add_argument('--derived-episodes',type=Path)
    p.add_argument('--reported-episodes',type=Path)
    p.add_argument('--clip-phase1',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    assert args.derived.exists() and args.derived_arrays.exists()
    # Freeze the independent result identity before any runner output is opened.
    identity=dict(record_sha256=sha_file(args.derived),arrays_sha256=sha_file(args.derived_arrays))
    own=json.loads(args.derived.read_text())
    c=Comparison()
    if args.clip_phase1:
        assert own['backbone']=='clip_b32'
        pass_path=T/'20261125_artelingo_held_test/results/seed42_pass_counts.json'
        report=json.loads(pass_path.read_text())
        compare_stats(c,own['stats'],report)
        picked=json.loads((T/'20261125_artelingo_held_test/results/picks_seed42.json').read_text())
        for name in ('B','B0'):
            c.value('picks.'+name,own['picks'][name],[picked[name][str(h)] for h in (0,1)])
        c.value('picks.rca_lambda',own['picks']['rca_lambda'],[picked['lambda_picks']['rca'][str(h)] for h in (0,1)])
        c.value('picks.aff',own['picks']['aff'],[39,119])
        c.value('picks.cf',own['picks']['cf'],[149,10])
        c.value('picks.sigma_star',own['picks']['sigma_star'],[0.,0.])
        c.value('picks.switched_off',own['picks']['switched_off'],False)
        tau=json.loads((T/'20261117_reader_fix_csd/results/rc_tau.json').read_text())['taus']
        c.value('picks.taus',own['picks']['taus'],tau)
        r3=T/'20261121_round3_affect_gate/results/seed42_arrays.npz'
        with np.load(args.derived_arrays) as arrays:
            required=[k for k in arrays.files if k.startswith(('aff_fused__','aff_cf__','aff_gate__','P__','margin__','pick__'))]
            required+=['cl','pair_index','parity','taus','aff_fused_cells','aff_cf_cells','aff_sigma']
        c.arrays(args.derived_arrays,r3,required)
    else:
        assert args.reported and args.reported_arrays, 'per-read agreement requires runner scalars AND arrays'
        report=json.loads(args.reported.read_text())
        compare_stats(c,own['stats'],report.get('stats',report))
        required=[f'{name}__{m}' for name in ('aff_fused','aff_cf','cosine','rca','B','B0')
                  for m in ('r1','gain','other','swap','strict')]+['cl','pair_index']
        if own.get('mode')!='selection': required.append('seed_index')
        c.arrays(args.derived_arrays,args.reported_arrays,required)
        if args.derived_episodes or args.reported_episodes:
            assert args.derived_episodes and args.reported_episodes
            c.arrays(args.derived_episodes,args.reported_episodes)
    result=c.record()
    result.update(independent_identity=identity,read=own.get('read','seed42'),backbone=own['backbone'])
    save_json(args.out,result)
    print('agreement:',result['agreement'],'compared:',result['compared'],'differences:',result['failed'])
    if not result['agreement']: raise SystemExit(1)


if __name__=='__main__': main()
