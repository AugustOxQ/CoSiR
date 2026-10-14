"""fx-rr: break each guard the new tests cover once in place, run the named test file, restore the file byte for byte.

python fx_rr_mutants.py   (from the round folder). Writes fx_rr_mutation_results.json.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
RUN = HERE / 'run_r9_descriptive.py'
CMP = HERE / 'rederive' / 'rd9_compare.py'
APPLY = HERE / 'run_r9_apply_rule.py'
PRE = HERE / 'run_r9_preread.py'
ST2 = HERE / 'run_r9_stage2.py'
T_DESC, T_CMP, T_APPLY, T_PRE, T_ST2 = ('test_r9_descriptive.py', 'rederive/rd9_test_compare.py', 'test_r9_apply_rule.py',
                                       'test_r9_preread.py', 'test_r9_stage2.py')

M = [  # name, file, old, new, test file
    # RR1 M1: the H5 copy
    ('h5_keys_check', RUN, "require(set(z.files) == set(want), 'H5 arrays keys')", "pass", T_DESC),
    ('h5_shape_check', RUN, "require(all(d[k].shape == (n,) for k in want), 'H5 arrays shape')", "pass", T_DESC),
    ('h5_index_dtype_check', RUN, "require(all(d[k].dtype == np.int64 for k in H5_EXTRA), 'H5 arrays index dtype')", "pass", T_DESC),
    ('h5_seed_order_check', RUN, "require(np.array_equal(d['seed_index'], np.repeat(np.arange(len(seeds), dtype=np.int64), n_seed)), 'H5 arrays seed order')", "pass", T_DESC),
    ('h5_pair_order_check', RUN, "require(np.array_equal(d['pair_index'][sl], pi_seed), 'H5 arrays pair order')", "pass", T_DESC),
    ('h5_float64_check', RUN, "x.dtype == np.float64 and bool(np.isfinite(x).all())", "True", T_DESC),
    ('h5_float64_only_dtype', RUN, "x.dtype == np.float64 and bool(np.isfinite(x).all())", "x.dtype == np.float64", T_DESC),
    ('h5_episode_count', RUN, "rec.get('n_episodes') == len(cl) and rec.get('n_clusters') == len(np.unique(cl))", "True", T_DESC),
    ('h5_cluster_count', RUN, "rec.get('n_episodes') == len(cl) and rec.get('n_clusters') == len(np.unique(cl))", "rec.get('n_episodes') == len(cl)", T_DESC),
    ('h5_n', RUN, "require(int(int_le0.sum()) == want['n'] and pc['point']", "require(pc['point']", T_DESC),
    ('h5_point', RUN, "pc['point'] == want['point'] and pc['ci95']", "pc['ci95']", T_DESC),
    ('h5_interval', RUN, "pc['ci95'] == list(want['ci95'])\n                and ST.QUANTITIES", "True\n                and ST.QUANTITIES", T_DESC),
    ('h5_quantity', RUN, "and ST.QUANTITIES[c][2] == want['quantity'], 'H5 arrays reproduce pass: '", ", 'H5 arrays reproduce pass: '", T_DESC),
    ('h5_secondary_skipped', RUN, "for c in ST.CHECKS + ST.SECONDARY:\n        want", "for c in ST.CHECKS:\n        want", T_DESC),
    ('h5_inputs_check_not_called', RUN, "    h5_check_pass(core, rec)\n    eps, inputs", "    eps, inputs", T_DESC),
    # RR1 N1: the selection replay does not compare margins, and says so
    ('cmp_selection_compares_margins', CMP, "compare_margins=not selection)", "compare_margins=True)", T_CMP),
    ('cmp_no_note', CMP, "            extra['margins_not_compared']=(", "            _unused=(", T_CMP),
    ('cmp_margins_never_compared', CMP, "if compare_margins and any(", "if False and any(", T_CMP),
    ('cmp_margins_skipped_in_every_mode', CMP, "compare_margins=not selection)", "compare_margins=False)", T_CMP),
    # RR1 N3: the agreement is bound to the sensitivity file
    ('apply_sensitivity_sha_not_checked', APPLY, 'if agr.get("sensitivity_sha256") != sens_sha:', 'if False:', T_APPLY),
    ('apply_sensitivity_sha_optional', APPLY, 'if agr.get("sensitivity_sha256") != sens_sha:', 'if "sensitivity_sha256" in agr and agr["sensitivity_sha256"] != sens_sha:', T_APPLY),
    ('preread_stand_in_without_sha', PRE, "           'sensitivity_sha256': R.sha256_file(out / names.sensitivity),   # the apply step binds it to its own file\n", "", T_PRE),
    # RR2 LOW-5
    ('stage2_prov_overlap', ST2, "    require(not set(made) & set(copied), 'made/copied keys overlap')\n", "", T_ST2),
]
out = {}
for name, path, old, new, test in M:
    raw = path.read_bytes()
    text = raw.decode()
    assert text.count(old) == 1, (name, text.count(old))
    path.write_text(text.replace(old, new))
    try:
        r = subprocess.run([PY, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', test],
                           cwd=HERE, capture_output=True, text=True, timeout=230)
    finally:
        path.write_bytes(raw)
    failed = [l for l in r.stdout.splitlines() if l.startswith('FAILED')]
    out[name] = {'caught': r.returncode != 0, 'first_failure': failed[:1]}
    print(name, out[name]['caught'], failed[:1], flush=True)
(HERE / 'fx_rr_mutation_results.json').write_text(json.dumps(out, indent=1) + '\n')
print('caught', sum(v['caught'] for v in out.values()), 'of', len(out))
