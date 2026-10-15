"""C13 agreement as R3 §8. Only opens runner outputs after our record exists.

Writes one JSON per read. Missing quantities are differences, never agreements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rd9_inputs import HERE,RD,T,pin_entries,read_pinned,save_json,sha_file,near_duplicates_entries,r6_sensitivity_entries,resolve_pin
from rd9_stats import CHECK_IDS,QUANTITIES


MISSING=object()    # a key absent from either record; never equal to anything, None included


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
        if ours is MISSING or theirs is MISSING:
            self.items.append(dict(key=key,equal=False,reason='missing key',
                                   ours=None if ours is MISSING else ours,
                                   theirs=None if theirs is MISSING else theirs));return
        if isinstance(ours,dict):
            if not isinstance(theirs,dict):
                self.items.append(dict(key=key,equal=False,reason='missing object'));return
            for k,v in ours.items(): self.value(f'{key}.{k}',v,theirs.get(k,MISSING))
        elif isinstance(ours,(list,tuple)):
            if not isinstance(theirs,(list,tuple)) or len(ours)!=len(theirs):
                self.items.append(dict(key=key,equal=False,reason='sequence length/type'));return
            if not ours:
                self.items.append(dict(key=key,equal=True,ours=ours,theirs=theirs));return
            for j,(a,b) in enumerate(zip(ours,theirs)): self.value(f'{key}.{j}',a,b)
        else:
            if theirs is MISSING: equal=False
            elif ours is None and theirs is None: equal=True    # present on both sides, both None (e.g. an untested level)
            elif ours is None or theirs is None: equal=False
            else: equal=compare_value(ours,theirs,key)
            self.items.append(dict(key=key,equal=bool(equal),ours=ours,theirs=None if theirs is MISSING else theirs,
                                   **(dict(reason='missing key') if theirs is MISSING else {})))
    def arrays(self,ours_path,theirs_path,required=None):
        with np.load(ours_path,allow_pickle=False) as ours,np.load(theirs_path,allow_pickle=False) as theirs:
            for key in required or ours.files:
                equal=key in ours.files and key in theirs.files and compare_array(ours[key],theirs[key])
                self.items.append(dict(key='arrays.'+key,equal=bool(equal)))
    def record(self):
        failures=[x for x in self.items if not x['equal']]
        assert self.items
        return dict(agreement=not failures,compared=len(self.items),failed=len(failures),items=self.items)


def compare_stats(comparison,ours,theirs,sensitivity=None,compare_margins=True):
    """`sensitivity`: the runner's sensitivity file {P1..P7: {SE,x,x95}}; compared when our record carries margins.
    compare_margins=False (the H8 selection replay): the margins are not compared, because R6's seed-42 file holds only the
    variance components, so there is no reported SE, x or x95 to compare with (the variance inputs are compared in phase 1)."""
    comparison.value('n_episodes',ours['n_episodes'],theirs.get('n_episodes',MISSING))
    comparison.value('n_clusters',ours['n_clusters'],theirs.get('n_clusters',MISSING))
    comparison.value('seeds',ours['seeds'],theirs.get('seeds',MISSING))
    comparison.value('episodes_sha256',ours['episodes_sha256'],theirs.get('episodes_sha256',MISSING))
    for j in CHECK_IDS:
        o=ours['checks'][j]
        t=theirs.get('checks',{}).get(j,{})
        comparison.value('checks.'+j+'.quantity',QUANTITIES[j],t.get('quantity',MISSING))
        if compare_margins and any(k in o for k in ('SE','x','x95')):
            # The detectable margin of rule §7.4: SE, x and x95 come from the runner's sensitivity file, not the pass file.
            margins=(sensitivity or {}).get(j,{})
            for k in ('SE','x','x95'): comparison.value('checks.'+j+'.'+k,o[k],margins.get(k,MISSING))
        for k in ('n','point','ci95','tested'):
            if k=='tested':
                # A seven-check record always tests all seven. A28 records must specify this flag.
                target=t.get('tested',True if 'holm_order' in theirs else MISSING)
            else:
                target=t.get(k,t.get('n_j',MISSING) if k=='n' else MISSING)
            comparison.value('checks.'+j+'.'+k,o[k],target)
        if o['tested']:
            for k in ('holm_k','ci_holm','passes','own_count_passes','near_boundary','level_two_sided'):
                alias={'passes':'pass','near_boundary':'within_one_of_boundary','level_two_sided':'holm_level_two_sided'}.get(k)
                comparison.value('checks.'+j+'.'+k,o[k],t.get(k,t.get(alias,MISSING) if alias else MISSING))
    # The runner's form: a list of seven, or {"level1": [...], "level2": [...]} for the two-level reads.
    if 'holm_order' in ours: comparison.value('holm_order',ours['holm_order'],theirs.get('holm_order',MISSING))


def derived_set_aside(own):
    """Flatten every seed/pair's exclusions; the outer seed is the read seed, not the builder seed."""
    records=own.get('episode_records',MISSING)
    if records is MISSING: return MISSING
    assert isinstance(records,dict)
    entries=[]
    for seed in own['stats']['seeds']:
        parts=records.get(str(seed),MISSING)
        if parts is MISSING: return MISSING
        assert isinstance(parts,dict)
        for pair in ('emotion__style','emotion__genre','style__genre'):
            part=parts.get(pair,MISSING)
            if part is MISSING: return MISSING
            assert isinstance(part,dict)
            excluded=part.get('set_aside',MISSING)
            if excluded is MISSING: return MISSING
            assert isinstance(excluded,list)
            for entry in excluded:
                assert isinstance(entry,dict)
                language=entry.get('language',MISSING)
                if language is MISSING: return MISSING
                assert isinstance(language,str) and type(seed) is int
                entries.append((language,pair,seed))
    return sorted(set(entries))


def reported_set_aside(value):
    if value is MISSING: return MISSING
    assert isinstance(value,list)
    entries=[]
    for entry in value:
        assert isinstance(entry,dict)
        if any(key not in entry for key in ('language','pair','seed')): return MISSING
        assert isinstance(entry['language'],str) and isinstance(entry['pair'],str) and type(entry['seed']) is int
        entries.append((entry['language'],entry['pair'],entry['seed']))
    return sorted(entries)    # a repeated entry stays: the derived side is a set, so the lengths then differ


def sorted_languages(value):
    if value is MISSING: return MISSING
    assert isinstance(value,list) and all(isinstance(language,str) for language in value)
    return sorted(value)


PASS_MODES={'held':False,'regression':False,'smoke':True}


def provenance(own,reported_path,raw,report):
    """Fields the apply-rule step reads from a per-read agreement file (ticket 19). Missing input crashes.

    smoke comes from the reported pass file's own mode field, held_pass_sha256 from its exact bytes,
    rederived_counts from the independent pass's n_j (the same keys for the two-level A28 reads).
    """
    top=report.get('mode')
    mode=top if top is not None else report.get('stats',{}).get('mode') if isinstance(report.get('stats'),dict) else None
    if mode not in PASS_MODES: raise ValueError(f'{reported_path.name}: mode field {mode!r} is not one of {sorted(PASS_MODES)}')
    checks=own['stats']['checks']
    if set(checks)!=set(CHECK_IDS): raise ValueError('derived record: checks are not exactly '+','.join(CHECK_IDS))
    counts={}
    for j in CHECK_IDS:
        n=checks[j]['n']
        if isinstance(n,bool) or not isinstance(n,int): raise ValueError(f'derived record: n of {j} is not an integer: {n!r}')
        counts[j]=n
    return dict(phase=2,smoke=PASS_MODES[mode],pass_file=reported_path.name,
                held_pass_sha256=hashlib.sha256(raw).hexdigest(),rederived_counts=counts)


def phase1_certificate(own,pins_path):
    """Bind the independently produced phase-1 inputs to the pins compared now."""
    from rd9_inputs import verify_reads
    assert own['inputs'].get(str(pins_path),{}).get('sha256')==sha_file(pins_path), 'phase-1 pins binding missing'  # guard:preread_compare_pins
    verify_reads(own['inputs'])  # guard:preread_compare_sources
    return dict(phase=1,pins_sha256=sha_file(pins_path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--derived',type=Path,required=True)
    p.add_argument('--reported',type=Path)
    p.add_argument('--derived-arrays',type=Path,required=True)
    p.add_argument('--reported-arrays',type=Path)
    p.add_argument('--reported-sensitivity',type=Path,help='sensitivity_<Hn>.json of the runner: SE, x, x95 per check')
    p.add_argument('--derived-episodes',type=Path)
    p.add_argument('--reported-episodes',type=Path)
    p.add_argument('--language-flags',type=Path,default=HERE/'results/language_flags_a28.json')
    p.add_argument('--near-duplicates',type=Path,default=HERE/'results/near_duplicates_a28.json')
    p.add_argument('--a28-pins',type=Path,default=HERE/'pins_a28.json')
    p.add_argument('--clip-phase1',action='store_true')
    p.add_argument('--phase1',action='store_true')
    p.add_argument('--backbone',choices=('clip_b32','qwen3vl_emb_2b','siglip2_so400m'))
    p.add_argument('--pins',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    assert args.derived.exists() and args.derived_arrays.exists()
    # Freeze the independent result identity before any runner output is opened.
    identity=dict(record_sha256=sha_file(args.derived),arrays_sha256=sha_file(args.derived_arrays))
    own=json.loads(args.derived.read_text())
    c=Comparison()
    extra={}
    if args.phase1 or args.clip_phase1:
        pins_path=args.pins or HERE/('pins_clip.json' if own['backbone']=='clip_b32' else f"pins_{own['backbone']}.json")
        extra.update(phase1_certificate(own,pins_path))
        extra['independent_files']={'record':str(args.derived.resolve()),'arrays':str(args.derived_arrays.resolve())}
    if args.phase1 and args.backbone!='clip_b32':
        from rd9_inputs import Inputs
        assert args.backbone==own['backbone'], 'phase-1 backbone mismatch'
        inputs=Inputs(args.backbone,args.pins)
        assert inputs.pin_record is not None, '--phase1 requires a pins file with seed42 record'
        with np.load(args.derived_arrays) as archive: arrays={k:archive[k] for k in archive.files}
        seed42=inputs.pin_record.get('seed42',inputs.pin_record.get('seed42_record',{}))
        # The seed-42 sigma file is not a pins part: the record names it ({path, sha256}), and it is read pinned.
        assert 'sensitivity_seed42' in inputs.pin_record, 'pins must name sensitivity_seed42'
        sigma_record=inputs.pin_record['sensitivity_seed42']
        sigma_path=resolve_pin(sigma_record['path'])
        sigma=json.loads(read_pinned(sigma_path,{sigma_record['path']:sigma_record['sha256']}))
        compare_phase1(c,own,arrays,inputs.json('cells_seed42'),inputs.json('picks_seed42'),
                       inputs.json('baselines_seed42'),inputs.json('rc_tau'),seed42,sigma)
    elif args.clip_phase1 or (args.phase1 and args.backbone=='clip_b32'):
        assert own['backbone']=='clip_b32'
        pass_path=T/'20261125_artelingo_held_test/results/seed42_pass_counts.json'
        report=json.loads(pass_path.read_text())
        compare_stats(c,own['stats'],report)
        sigma_path=T/'20261125_artelingo_held_test/results/sensitivity_seed42.json'
        compare_sigma(c,own,json.loads(read_pinned(sigma_path,r6_sensitivity_entries())))
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
            for name in ('B','B0','B1'):
                values=arrays.get(name+'__r1')
                got=None if values is None else 100*float(np.mean(values))
                c.value('seed42.mean_r1.'+name,got,picked.get('mean_r1',{}).get(name))
            # The floor is explicit: a derivation that omits an array must disagree, not shrink the comparison.
            floor=[f'{arm}__{metric}' for arm in ('aff_fused','aff_cf') for metric in ('r1','gain','other','swap','strict')]
            floor+=[f'{prefix}__{side}' for prefix in ('aff_gate','P','margin','pick') for side in ('a','b')]
            floor+=['cl','pair_index','parity','taus','aff_fused_cells','aff_cf_cells','aff_sigma']
            required=sorted(set(floor)|{k for k in arrays.files if k.startswith(('aff_fused__','aff_cf__','aff_gate__','P__','margin__','pick__'))})
        c.arrays(args.derived_arrays,r3,required)
    else:
        assert args.reported and args.reported_arrays, 'per-read agreement requires runner scalars AND arrays'
        raw=args.reported.read_bytes()
        report=json.loads(raw)
        extra=provenance(own,args.reported,raw,report)
        sensitivity=None
        if args.reported_sensitivity:
            sensitivity_raw=args.reported_sensitivity.read_bytes()
            sensitivity=json.loads(sensitivity_raw)
            extra['sensitivity_sha256']=hashlib.sha256(sensitivity_raw).hexdigest()
        selection=own.get('mode')=='selection'
        if selection:
            extra['margins_not_compared']=('SE, x and x95 not compared, selection replay: the reported side is round 6 seed-42 pass counts, whose sensitivity '
                                           'file holds only the variance components (no SE, x or x95); the variance inputs are '
                                           'compared in phase 1')
        compare_stats(c,own['stats'],report.get('stats',report),sensitivity,compare_margins=not selection)
        if own.get('read') in ('H8','H9','H10') and own.get('mode')!='selection':
            c.value('set_aside',derived_set_aside(own),reported_set_aside(report.get('set_aside',MISSING)))
            # The flags file must be the pinned stage-1 file. The near-duplicates file is in no pins file; its SHA-256 is
            # pinned in rd9_inputs and recorded. Each file is parsed from the bytes that were hashed.
            flags_raw=read_pinned(args.language_flags,pin_entries(json.loads(args.a28_pins.read_text())))
            duplicates_raw=read_pinned(args.near_duplicates,near_duplicates_entries())
            extra.update(language_flags_sha256=hashlib.sha256(flags_raw).hexdigest(),
                         near_duplicates_sha256=hashlib.sha256(duplicates_raw).hexdigest())
            flags=json.loads(flags_raw)
            assert isinstance(flags,dict), 'language flags must be an object'
            c.value('flagged_languages',sorted_languages(flags.get('flagged',MISSING)),
                    sorted_languages(report.get('flagged_languages',MISSING)))
            duplicates=json.loads(duplicates_raw)
            assert isinstance(duplicates,dict), 'near duplicates must be an object'
            assert duplicates.get('train_side')==912, 'near duplicates: train_side must be 912'
            assert duplicates.get('read_paintings')==746, 'near duplicates: read_paintings must be 746'
            reported_duplicates=report.get('near_duplicates',{})
            assert isinstance(reported_duplicates,dict), 'reported near duplicates must be an object'
            for key in ('cos_099','cos_098'):
                a,b=duplicates.get(key,MISSING),reported_duplicates.get(key,MISSING)
                for value in (a,b):
                    if value is not MISSING:
                        assert type(value) is int and 0<=value<=746, 'near duplicates: counts must be integers in 0..746'
                c.value('near_duplicates.'+key,a,b)
        required=[f'{name}__{m}' for name in ('aff_fused','aff_cf','cosine','rca','B','B0')
                  for m in ('r1','gain','other','swap','strict')]+['cl','pair_index']
        if own.get('mode')!='selection': required.append('seed_index')
        c.arrays(args.derived_arrays,args.reported_arrays,required)
        if args.derived_episodes or args.reported_episodes:
            assert args.derived_episodes and args.reported_episodes
            c.arrays(args.derived_episodes,args.reported_episodes)
    result=c.record()
    result.update(independent_identity=identity,read=own.get('read','seed42'),backbone=own['backbone'],**extra)
    save_json(args.out,result)
    print('agreement:',result['agreement'],'compared:',result['compared'],'differences:',result['failed'])
    if not result['agreement']: raise SystemExit(1)




def compare_sigma(c,own,sigma):
    """Seed-42 sensitivity inputs (R6 rule §8.4): the variance components of the seven differences, N and the clusters."""
    sigma=sigma or {}
    stats=own['stats']
    c.value('seed42.sigma.N',stats['n_episodes'],sigma.get('N',MISSING))
    c.value('seed42.sigma.n_paintings',stats['n_clusters'],sigma.get('n_paintings',MISSING))
    variance=stats.get('sensitivity',{})
    for j in CHECK_IDS:
        for ours_key,theirs_key in (('sigma_a2','sigma_a2'),('sigma_e2','sigma_eps2')):
            c.value(f'seed42.sigma.{j}.{ours_key}',variance.get(j,{}).get(ours_key,MISSING),
                    sigma.get(j,{}).get(theirs_key,MISSING))


def compare_phase1(c,own,arrays,cells,picks,baselines,tau,seed42,sigma=None):
    """Compare independently computed quantities against data pins, including switched-off runs."""
    from rd9_inputs import sha_bytes
    for key in ('aff','cf','sigma_star','switched_off','gate_share'):
        c.value('picks.'+key,own['picks'][key],cells.get(key))
    for key in ('B','B0'):
        v=picks.get(key)
        if isinstance(v,dict): v=[v.get(str(h)) for h in (0,1)]
        c.value('picks.'+key,own['picks'][key],v)
    lambdas=baselines.get('lambda_picks',{}).get('rca')
    if lambdas is None:
        lambdas=baselines.get('scorers',{}).get('rca',{}).get('lambda_picks')
    if isinstance(lambdas,dict): lambdas=[lambdas.get(str(h)) for h in (0,1)]
    c.value('picks.rca_lambda',own['picks']['rca_lambda'],lambdas)
    c.value('picks.taus',own['picks']['taus'],tau.get('taus'))
    for name,arm in (('B','B'),('B0','B0'),('B1','B1'),('AFF','aff_fused'),('CF','aff_cf')):
        got=100*float(np.mean(arrays[arm+'__r1'])) if arm+'__r1' in arrays else None
        c.value('seed42.mean_r1.'+name,got,seed42.get('mean_r1',{}).get(name))
    for j in CHECK_IDS:
        target=seed42.get('checks',{}).get(j,{})
        c.value('seed42.checks.'+j+'.point',own['stats']['checks'][j]['point'],target.get('point'))
        c.value('seed42.checks.'+j+'.n',own['stats']['checks'][j]['n'],target.get('n_j',target.get('n')))
    compare_sigma(c,own,sigma)
    hashes=seed42.get('array_hashes',{})
    if not hashes and 'arrays' in seed42:
        # Stage 2 records the 96 stage-0a keys with their archive namespaces.
        # Phase 1 independently exposes the primary scorers and reader inputs;
        # R1 and PM belong to the post-verdict phase, not this comparison.
        hashes={}
        for key,entry in seed42['arrays'].items():
            prefix,_,name=key.partition('/')
            if prefix=='per_anchor_seed42' and name=='anchor_group': name='cl'
            if name in arrays: hashes[name]=entry
        required={f'{arm}__{metric}' for arm in ('aff_fused','aff_cf','cosine','rca')
                  for metric in ('r1','gain','other','swap','strict')}
        required|={'cl','pair_index','parity','aff_fused_cells','aff_cf_cells'}
        required|={f'{prefix}__{side}' for prefix in ('aff_gate','P','margin','pick') for side in ('a','b')}
    else:
        required={f'{arm}__{metric}' for arm in ('B','B0','aff_fused','aff_cf','cosine','rca')
                  for metric in ('r1','gain','other','swap','strict')}
    # Every recorded per-anchor array, plus all primary scorer arrays, must be present.
    assert hashes, 'pins must include seed42.array_hashes'
    for key in sorted(set(hashes)|required):
        entry=hashes.get(key)
        if entry is None:
            c.value('seed42.arrays.'+key+'.present',True,False);continue
        v=arrays.get(key)
        observed=None if v is None else {'sha256':sha_bytes(v),'dtype':str(v.dtype),'shape':list(v.shape)}
        c.value('seed42.arrays.'+key,observed,entry)


if __name__=='__main__': main()
