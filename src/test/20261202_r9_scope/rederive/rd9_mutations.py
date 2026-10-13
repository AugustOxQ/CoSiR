"""Scoped, one-line mutations: run their discriminating pytest case and restore in finally.

Every modified file belongs to this ticket. No implementation-chain files are used.
A timeout or an infrastructure error is not counted as a killed mutation.
"""
import argparse
import os
import subprocess
from pathlib import Path
from rd9_inputs import RD,OUT,save_json

CASES=[]
def case(name,module,before,after,test):
    CASES.append((name,module,before,after,test))
A='rd9_test_arithmetic.py::'
E='rd9_test_a28.py::'
C='rd9_test_compare.py::'
case('P5 uses CF','rd9_stats.py',"'P5':('r1','aff_cf')","'P5':('r1','B')",A+'test_pass_record_nonzero_counts_comparators_zero_inclusion_and_holm_level')
case('P7 uses RCA gain','rd9_stats.py',"'P7':('gain','rca')","'P7':('gain','aff_cf')",A+'test_pass_record_nonzero_counts_comparators_zero_inclusion_and_holm_level')
case('zero integer sums count','rd9_stats.py','np.count_nonzero(ints<=0)','np.count_nonzero(ints<0)',A+'test_pass_record_nonzero_counts_comparators_zero_inclusion_and_holm_level')
case('Holm interval quantiles','rd9_stats.py','[100*alpha,100*(1-alpha)]','[2.5,97.5]',A+'test_pass_record_nonzero_counts_comparators_zero_inclusion_and_holm_level')
case('bootstrap seed 42','rd9_stats.py','5000, 42, 250','5000, 43, 250',A+'test_pass_record_nonzero_counts_comparators_zero_inclusion_and_holm_level')
case('switched off needs both','rd9_scores.py','switched_off=all(','switched_off=any(',A+'test_sigma_ties_choose_smallest_and_switched_off_requires_both_halves')
case('sigma ascending tie order','rd9_scores.py','sums=sorted({u+a for u in NESTED_U for a in NESTED_A})','sums=sorted({u+a for u in NESTED_U for a in NESTED_A},reverse=True)',A+'test_sigma_ties_choose_smallest_and_switched_off_requires_both_halves')
case('sigma first argmax','rd9_scores.py','sigma_idx=control.argmax(0)','sigma_idx=control.shape[0]-1-control[::-1].argmax(0)',A+'test_sigma_ties_choose_smallest_and_switched_off_requires_both_halves')
case('B first tied cell','rd9_scores.py','if rho[h]>best[h]:','if rho[h]>=best[h]:',A+'test_baseline_ties_choose_first_cell')
case('gate inclusive margin','rd9_scores.py','margins[c]>=tau','margins[c]>tau',A+'test_reader_tau_percentiles_inclusive_gate_and_affect_pick')
case('gate affect restriction','rd9_scores.py','(picks[c]==0)','(picks[c]==1)',A+'test_reader_tau_percentiles_inclusive_gate_and_affect_pick')
case('opposite parity','rd9_scores.py','rows = parity == 1 - h','rows = parity == h',A+'test_assembly_uses_opposite_parity')
case('tau 75th percentile','rd9_scores.py','[0,25,50,75]','[0,25,50,100]',A+'test_reader_tau_percentiles_inclusive_gate_and_affect_pick')
case('RCA extension trigger','rd9_scores.py','if best==16.:','if best==8.:',A+'test_rca_extends_grid_after_sixteen')
case('RCA extends to 32','rd9_scores.py','candidates+[32.,64.]','candidates+[64.]',A+'test_rca_extends_grid_after_sixteen')
case('RCA extends to 64','rd9_scores.py','candidates+[32.,64.]','candidates+[32.]',A+'test_rca_extends_grid_after_sixteen')
case('RCA extension first tie','rd9_scores.py','candidates+[32.,64.]','candidates+[64.,32.]',A+'test_rca_extends_grid_after_sixteen')
case('language seed multiplier','rd9_episodes.py','1000 * seed + LANGUAGES.index(lang) + 1','100 * seed + LANGUAGES.index(lang) + 1',E+'test_a28_seeds_post_drop_shares_and_language_order')
case('canonical language index','rd9_episodes.py','LANGUAGES.index(lang) + 1','remaining.index(lang) + 1',E+'test_a28_seeds_post_drop_shares_and_language_order')
case('post-drop shares','rd9_episodes.py','sizes[lang] = len(pool_rows(sub, rows))','sizes[lang] = len(pool_rows(sub, np.arange(len(R))))',E+'test_a28_seeds_post_drop_shares_and_language_order')
case('kept rows only','rd9_episodes.py','rows = np.flatnonzero(keep[R])','rows = np.arange(len(R))',E+'test_a28_seeds_post_drop_shares_and_language_order')
case('pre-drop value index','rd9_episodes.py','indices[lang] = PaintingValueIndex(sub, groups[R])','indices[lang] = PaintingValueIndex({a:v[rows] for a,v in sub.items()}, groups[R][rows])',E+'test_a28_dropped_labels_still_count_for_lacks_and_index_precedes_drops')
case('global row mapping','rd9_episodes.py','*(R[getattr(part,f)] for f in FIELDS)','*(getattr(part,f) for f in FIELDS)',E+'test_a28_seeds_post_drop_shares_and_language_order')
case('language-order assembly','rd9_episodes.py','combined = concat(parts,a,b)','combined = concat(parts[::-1],a,b)',E+'test_a28_seeds_post_drop_shares_and_language_order')
case('values subset guard','rd9_episodes.py','if not set(wanted) <= set(valid):','if False:',E+'test_restricted_subset_guard_and_minimum_paintings')
case('minimum passed to primitive','rd9_episodes.py','min_paintings=min_paintings, index=index','min_paintings=1, index=index',E+'test_restricted_subset_guard_and_minimum_paintings')
case('A28 minimum is one','rd9_episodes.py','idx,(codes[a],codes[b]),min_paintings=1','idx,(codes[a],codes[b]),min_paintings=30',E+'test_a28_dropped_labels_still_count_for_lacks_and_index_precedes_drops')
case('subset set-aside catch','rd9_episodes.py','except ValuesSubsetError:','except KeyError:',E+'test_a28_both_set_aside_causes_rebuild_with_new_shares')
case('failed-draw catch','rd9_episodes.py','except RuntimeError as error:','except ValueError as error:',E+'test_a28_both_set_aside_causes_rebuild_with_new_shares')
case('re-split remaining languages','rd9_episodes.py',"remaining = [l for l in remaining if l not in {f['language'] for f in failed}]","remaining = [l for l in remaining if l in {f['language'] for f in failed}]",E+'test_a28_both_set_aside_causes_rebuild_with_new_shares')
case('record set-asides','rd9_episodes.py','excluded.extend(failed)','excluded.extend([])',E+'test_a28_both_set_aside_causes_rebuild_with_new_shares')
case('third set-aside stops','rd9_episodes.py','assert len(excluded) <= 2','assert len(excluded) <= 3',E+'test_a28_third_set_aside_stops')
case('hash within-read uniqueness','rd9_phase2.py',"assert len(current)==len(set(current)), 'repeated episode hash within this read'","assert True, 'repeated episode hash within this read'",E+'test_episode_hash_guard_rejects_duplicates_and_prior_collisions')
case('hash prior collision','rd9_phase2.py',"assert not set(current)&set(prior), 'episode hash repeats a prior non-smoke episode'","assert True, 'episode hash repeats a prior non-smoke episode'",E+'test_episode_hash_guard_rejects_duplicates_and_prior_collisions')
case('phase1 switched-off comparison','rd9_compare.py',"('aff','cf','sigma_star','switched_off','gate_share')","('aff','cf','sigma_star','gate_share')",A+'test_phase1_agreement_checks_every_pinned_quantity')
case('phase1 n_j comparison','rd9_compare.py',"target.get('n_j',target.get('n'))","target.get('point')",A+'test_phase1_agreement_checks_every_pinned_quantity')
case('phase1 array hash comparison','rd9_compare.py',"'sha256':sha_bytes(v)","'sha256':entry['sha256']",A+'test_phase1_agreement_checks_every_pinned_quantity')
case('phase1 B1 mean comparison','rd9_compare.py',"('B1','B1'),",'',A+'test_phase1_agreement_checks_every_pinned_quantity')
case('pin-name ambiguity','rd9_inputs.py','assert len(candidates)<=1','assert len(candidates)<=2',A+'test_pin_roles_reject_ambiguous_names_and_choose_backbone_recipe')
case('backbone recipe preference','rd9_inputs.py','if preferred: candidates=preferred','if False: candidates=preferred',A+'test_pin_roles_reject_ambiguous_names_and_choose_backbone_recipe')
case('round9 paths use checkout','rd9_inputs.py','return HERE.parents[2]/path','return MAIN/path',A+'test_pin_paths_keep_round9_outputs_in_the_running_worktree')
case('library source hash','rd9_inputs.py','result[relative]=sha_file(path)',"result[relative]='0'*64",A+'test_library_provenance_is_sha256_of_loaded_src_files')
case('claim set-aside comparison','rd9_compare.py',
     "c.value('set_aside',derived_set_aside(own),reported_set_aside(report.get('set_aside',MISSING)))",
     'pass',C+'test_set_aside_difference_fails_agreement')
case('set-aside read seed mapping','rd9_compare.py','entries.append((language,pair,seed))',
     "entries.append((language,pair,entry['seed']))",C+'test_set_aside_compares_language_pair_and_read_seed_as_a_set')
case('missing derived set-asides','rd9_compare.py',"part.get('set_aside',MISSING)",
     "part.get('set_aside',[])",C+'test_missing_set_aside_fails_even_when_empty')
case('claim language flags comparison','rd9_compare.py',
     "c.value('flagged_languages',sorted_languages(flags.get('flagged',MISSING)),\n                    sorted_languages(report.get('flagged_languages',MISSING)))",
     'pass',C+'test_flagged_language_difference_fails_agreement')
case('claim language flags sorted','rd9_compare.py','return sorted(value)','return value',
     C+'test_flagged_languages_compare_as_sorted_lists')
case('claim near duplicates cos_099','rd9_compare.py',"for key in ('cos_099','cos_098'):",
     "for key in ('cos_098',):",C+'test_near_duplicate_difference_fails_agreement')
case('claim near duplicates cos_098','rd9_compare.py',"for key in ('cos_099','cos_098'):",
     "for key in ('cos_099',):",C+'test_near_duplicate_difference_fails_agreement')
case('near duplicates train-side population','rd9_compare.py',"duplicates.get('train_side')==912",'True',
     C+'test_near_duplicate_population_is_asserted')
case('near duplicates read population','rd9_compare.py',"duplicates.get('read_paintings')==746",'True',
     C+'test_near_duplicate_population_is_asserted')
case('missing either comparison side','rd9_compare.py','if ours is MISSING or theirs is MISSING:',
     'if theirs is MISSING:',C+'test_missing_either_side_is_a_serializable_difference')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT/'rd9_review_mutations.json')
    args=parser.parse_args()
    assert not args.out.exists(), f'refusing overwrite: {args.out}'
    results=[]
    for name,module,before,after,test in CASES:
        path=RD/module;original=path.read_text()
        assert original.count(before)==1,(name,original.count(before))
        try:
            path.write_text(original.replace(before,after,1))
            run=subprocess.run(['/root/miniconda3/envs/CoSiR/bin/python','-m','pytest','-q',str(RD/test),
                                '-o',f'cache_dir={OUT}/rd9_pytest_cache','--basetemp',str(OUT/'rd9_test_temp')],
                               env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','CUDA_VISIBLE_DEVICES':''},
                               capture_output=True,text=True,timeout=30)
            killed=run.returncode==1 and (' failed' in run.stdout or 'ERROR' in run.stdout)
            status='killed' if killed else 'survived' if run.returncode==0 else 'error'
            results.append(dict(mutation=name,module=module,before=before,after=after,test=test,status=status))
            print(name+': '+status,flush=True)
        finally:
            path.write_text(original)
    save_json(args.out,results)
    print('killed',sum(r['status']=='killed' for r in results),'survived',sum(r['status']=='survived' for r in results),'errors',sum(r['status']=='error' for r in results))
    assert all(r['status']=='killed' for r in results)


if __name__=='__main__': main()
