"""fx-c13: break each new guard once, run the named tests, restore the file (SHA-256 checked). Writes fx_c13_mutations.json.

Run from the worktree: python src/test/20261202_r9_scope/fx_c13_mutations.py [first_index last_index]
"""
import hashlib, json, subprocess, sys
from pathlib import Path
F = Path(__file__).resolve().parent
R = F / 'rederive'
PY = '/root/miniconda3/envs/CoSiR/bin/python'
P1, CMP = 'rd9_test_phase1.py', 'rd9_test_compare.py'
M = []
def m(name, file, old, new, tests, cwd=R, count=1):
    M.append(dict(name=name, file=file, old=old, new=new, tests=tests, cwd=cwd, count=count))
C = 'rd9_compare.py'
m('p1_sigma_compare_deleted', C, "    compare_sigma(c,own,sigma)\n    hashes=", "    hashes=", [P1])
m('clip_sigma_compare_deleted', C, "        compare_sigma(c,own,json.loads(read_pinned(sigma_path,r6_sensitivity_entries())))\n", "", [P1])
m('sigma_N_deleted', C, "    c.value('seed42.sigma.N',stats['n_episodes'],sigma.get('N',MISSING))\n", "", [P1])
m('sigma_clusters_deleted', C, "    c.value('seed42.sigma.n_paintings',stats['n_clusters'],sigma.get('n_paintings',MISSING))\n", "", [P1])
m('sigma_e2_not_compared', C, "(('sigma_a2','sigma_a2'),('sigma_e2','sigma_eps2'))", "(('sigma_a2','sigma_a2'),)", [P1])
m('sigma_a2_not_compared', C, "(('sigma_a2','sigma_a2'),('sigma_e2','sigma_eps2'))", "(('sigma_e2','sigma_eps2'),)", [P1])
m('cells_loop_aff_cf_sigma_star', C, "for key in ('aff','cf','sigma_star','switched_off','gate_share'):", "for key in ('switched_off','gate_share'):", [P1])
m('cells_gate_share', C, "for key in ('aff','cf','sigma_star','switched_off','gate_share'):", "for key in ('aff','cf','sigma_star','switched_off'):", [P1])
m('B_B0_picks_deleted', C, "    for key in ('B','B0'):\n        v=picks.get(key)", "    for key in ():\n        v=picks.get(key)", [P1])
m('B0_pick_only', C, "    for key in ('B','B0'):\n        v=picks.get(key)", "    for key in ('B',):\n        v=picks.get(key)", [P1])
m('rca_lambda_deleted', C, "    c.value('picks.rca_lambda',own['picks']['rca_lambda'],lambdas)\n", "", [P1])
m('taus_deleted', C, "    c.value('picks.taus',own['picks']['taus'],tau.get('taus'))\n", "", [P1])
m('tau_tolerance_exact', C, "return bool(np.isclose(ours,theirs,rtol=1e-9,atol=1e-15))", "return ours==theirs", [P1])
m('tau_tolerance_loose', C, "return bool(np.isclose(ours,theirs,rtol=1e-9,atol=1e-15))", "return bool(np.isclose(ours,theirs,rtol=1e-3))", [P1])
m('picks_exact_branch_removed', C, "    if key.startswith('picks.') and 'taus' not in key:\n        return ours==theirs\n", "", [P1])
m('mean_loop_B_only', C, "for name,arm in (('B','B'),('B0','B0'),('B1','B1'),('AFF','aff_fused'),('CF','aff_cf')):", "for name,arm in (('B','B'),):", [P1])
m('mean_B1_deleted', C, "('B1','B1'),", "", [P1])
m('P_points_deleted', C, "        c.value('seed42.checks.'+j+'.point',own['stats']['checks'][j]['point'],target.get('point'))\n", "", [P1])
m('P_n_deleted', C, "        c.value('seed42.checks.'+j+'.n',own['stats']['checks'][j]['n'],target.get('n_j',target.get('n')))\n", "", [P1])
m('floor_cl_etc', C, "        required|={'cl','pair_index','parity','aff_fused_cells','aff_cf_cells'}\n", "", [P1])
m('floor_reader_arrays', C, "        required|={f'{prefix}__{side}' for prefix in ('aff_gate','P','margin','pick') for side in ('a','b')}\n", "", [P1])
m('floor_scorer_arrays', C, "required={f'{arm}__{metric}' for arm in ('aff_fused','aff_cf','cosine','rca')", "required={f'{arm}__{metric}' for arm in ('aff_fused',)", [P1])
m('present_check_removed', C, "            c.value('seed42.arrays.'+key+'.present',True,False);continue", "            continue", [P1])
m('clip_B_means_deleted', C, "            for name in ('B','B0','B1'):\n                values=arrays.get(name+'__r1')", "            for name in ():\n                values=arrays.get(name+'__r1')", [P1])
m('clip_B1_mean_deleted', C, "            for name in ('B','B0','B1'):\n                values=arrays.get(name+'__r1')", "            for name in ('B','B0'):\n                values=arrays.get(name+'__r1')", [P1])
m('clip_floor_explicit_removed', C, "required=sorted(set(floor)|{k for k", "required=sorted({k for k", [P1])
m('clip_floor_taus_etc', C, "            floor+=['cl','pair_index','parity','taus','aff_fused_cells','aff_cf_cells','aff_sigma']\n", "", [P1])
m('compare_stats_margins_deleted', C, "            for k in ('SE','x','x95'): comparison.value('checks.'+j+'.'+k,o[k],margins.get(k,MISSING))\n", "            pass\n", [CMP])
m('margin_x_only_SE', C, "            for k in ('SE','x','x95'): comparison.value", "            for k in ('SE',): comparison.value", [CMP])
m('margin_x95_not_compared', C, "            for k in ('SE','x','x95'): comparison.value", "            for k in ('SE','x'): comparison.value", [CMP])
m('quantity_deleted', C, "        comparison.value('checks.'+j+'.quantity',QUANTITIES[j],t.get('quantity',MISSING))\n", "", [CMP])
m('set_aside_dup_dedupe', C, "    return sorted(entries)    # a repeated", "    return sorted(set(entries))    # a repeated", [CMP])
m('near_dup_unpinned', C, "duplicates_raw=read_pinned(args.near_duplicates,near_duplicates_entries())", "duplicates_raw=args.near_duplicates.read_bytes()", [CMP])
m('sensitivity_sha_not_recorded', C, "            extra['sensitivity_sha256']=hashlib.sha256(sensitivity_raw).hexdigest()\n", "            pass\n", [CMP])
m('sigma_pin_unchecked', C, "sigma=json.loads(read_pinned(sigma_path,{sigma_record['path']:sigma_record['sha256']}))", "sigma=json.loads(sigma_path.read_bytes())", [P1])
I = 'rd9_inputs.py'
m('parts_only_removed', I, "entries = parts_entries(self.pin_record) if 'parts' in self.pin_record else pin_entries(self.pin_record)", "entries = pin_entries(self.pin_record)", [P1])
m('named_roles_removed', I, "if 'parts' in self.pin_record and r in self.pin_record['parts'] and r not in FROZEN}", "if False}", [P1])
m('unpickle_check_removed', I, "    assert not imported, f'unpickling imported", "    assert True, f'unpickling imported", [P1])
m('unpickle_not_used_by_readers', I, "reader=unpickle(self.path('rb_reader_A0'", "reader=pickle.loads(self.path('rb_reader_A0'", [P1])
m('near_dup_constant_changed', I, "NEAR_DUPLICATES_SHA256 = 'ba92530d", "NEAR_DUPLICATES_SHA256 = 'ca92530d", [CMP])
m('near_pin_unchecked_in_held', F / 'run_r9_held.py', "    _require(R.sha256_bytes(near_raw) == NEAR_DUPLICATES_SHA256,", "    _require(True,", ['../test_r9_held_a28.py'])
m('held_pin_constant_changed', F / 'run_r9_held.py', 'NEAR_DUPLICATES_SHA256 = "ba92530d', 'NEAR_DUPLICATES_SHA256 = "ca92530d', ['../test_r9_held_a28.py'])
m('phase2_fits_unpickle_plain', 'rd9_phase2.py', "fits=unpickle(Path(", "fits=pickle.loads(Path(", [P1])

def run(mut):
    path = Path(mut['file']) if isinstance(mut['file'], Path) else R / mut['file']
    original = path.read_bytes()
    text = original.decode()
    assert text.count(mut['old']) == mut['count'], (mut['name'], text.count(mut['old']))
    sha = hashlib.sha256(original).hexdigest()
    try:
        path.write_text(text.replace(mut['old'], mut['new']))
        if mut['tests'] is None:
            return dict(name=mut['name'], result='not run (no unit test)')
        p = subprocess.run([PY, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', *mut['tests']], cwd=R,
                           capture_output=True, text=True, timeout=200,
                           env={'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin'})
        tail = [l for l in p.stdout.splitlines() if l.startswith(('FAILED', 'ERROR')) or ' passed' in l or ' failed' in l][-2:]
        return dict(name=mut['name'], caught=p.returncode != 0, detail=tail)
    finally:
        path.write_bytes(original)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha

if __name__ == '__main__':
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, len(M))
    out = F / ('fx_c13_mutations.json' if (lo, hi) == (0, len(M)) else f'fx_c13_mutations_part_{lo}_{hi}.json')
    results = [run(x) for x in M[lo:hi]]
    out.write_text(json.dumps(results, indent=1) + '\n')
    for r in results: print(r['name'], r.get('caught', r.get('result')))
