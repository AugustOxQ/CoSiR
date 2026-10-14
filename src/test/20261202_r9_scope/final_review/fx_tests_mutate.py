"""Fix wave fx-tests: guard mutations (adapted from fr_b2_mutate.py; worktree r9-fx-tests). Each mutation edits one source file of the snapshot (exactly one match
required), runs the named tests, restores the original bytes (checked by SHA-256) and records whether a test failed.
Usage: python fr_b2_mutate.py ID [ID ...]   (results appended to fr_b2_mutate.json)"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
F = Path("/project/CoSiR-wt/r9-fx-tests/src/test/20261202_r9_scope")
PY = "/root/miniconda3/envs/CoSiR/bin/python"
T0A, T0B, T2, TP, TD = ("test_r9_stage0a.py", "test_r9_stage0b.py", "test_r9_stage2.py", "test_r9_pins.py",
                        "test_r9_data.py")
M = {
 # stage 0a
 "0a_run_specific_extra": ("run_r9_stage0a.py", '"time", "input_sha256")   # rule 3.1', '"time", "input_sha256", "coef_sha256")   # rule 3.1', [T0A]),
 "0a_array_item_dtype": ("run_r9_stage0a.py", "g.shape == w.shape and g.dtype == w.dtype and np.array_equal(g, w)", "g.shape == w.shape and np.array_equal(g, w)", [T0A]),
 "0a_array_item_shape": ("run_r9_stage0a.py", "g.shape == w.shape and g.dtype == w.dtype and np.array_equal(g, w)", "g.dtype == w.dtype and np.array_equal(g.ravel(), w.ravel())", [T0A]),
 "0a_order": ("run_r9_stage0a.py", "if not diff and list(got) != list(want):", "if False:", [T0A]),
 "0a_item_fields": ("run_r9_stage0a.py", "        elif got[k] != want[k]:\n            fields", "        elif False:\n            fields", [T0A]),
 "0a_item_equal_flag": ("run_r9_stage0a.py", 'if got[k].get("equal") is not True:', 'if False:', [T0A]),
 "0a_items_missing": ("run_r9_stage0a.py", 'diff = [f"{k}: missing" for k in want if k not in got]', 'diff = []', [T0A]),
 "0a_items_extra": ("run_r9_stage0a.py", 'diff += [f"{k}: not in the reference" for k in got if k not in want]\n    for k in want:', 'for k in want:', [T0A]),
 "0a_counts_extra": ("run_r9_stage0a.py", 'diff += [f"{k}: not in the reference" for k in got if k not in want and k not in RUN_SPECIFIC]', 'pass', [T0A]),
 "0a_guard_pins": ("run_r9_stage0a.py", "_require(pins == PINS.build_pins(BACKBONE),", "_require(True or pins == PINS.build_pins(BACKBONE),", [T0A]),
 "0a_guard_cells": ("run_r9_stage0a.py", '                 list(S.CELLS["r1"]["cf"])], "r9_score.CELLS', '                 list(S.CELLS["r1"]["cf"])] or True, "r9_score.CELLS', [T0A]),
 # stage 0b
 "0b_array_diff_dtype": ("run_r9_stage0b.py", "    if got.dtype != want.dtype:\n        return [f\"{name}: dtype", "    if False:\n        return [f\"{name}: dtype", [T0B]),
 "0b_array_diff_nan": ("run_r9_stage0b.py", "np.array_equal(got, want, equal_nan=True) if got.dtype.kind == \"f\" else np.array_equal(got, want)", "np.allclose(got, want, equal_nan=True)", [T0B]),
 "0b_realpath": ("run_r9_stage0b.py", "if os.path.realpath(path) == os.path.realpath(pinned[role]):", "if False:", [T0B]),
 "0b_decide_regression": ("run_r9_stage0b.py", 'return bool(regression if case == "case1" else item3 and independent)', 'return bool(True if case == "case1" else item3 and independent)', [T0B]),
 "0b_decide_picks": ("run_r9_stage0b.py", "if not (item1 and gates and item4 and picks):", "if not (item1 and gates and item4):", [T0B]),
 "0b_decide_item4": ("run_r9_stage0b.py", "if not (item1 and gates and item4 and picks):", "if not (item1 and gates and picks):", [T0B]),
 "0b_decide_independent": ("run_r9_stage0b.py", 'return bool(regression if case == "case1" else item3 and independent)', 'return bool(regression if case == "case1" else item3)', [T0B]),
 "0b_decide_case3": ("run_r9_stage0b.py", 'if case not in ("case1", "case2"):', 'if case not in ("case1", "case2", "case3"):', [T0B]),
 "0b_routed": ("run_r9_stage0b.py", "# guard:routed_not_pinned", "# guard:routed_not_pinned\n        pass", [T0B]),
 "0b_npz_extra_key": ("run_r9_stage0b.py", 'out = [f"key only in the replay: {k}" for k in a.files if k not in b.files]', 'out = []', [T0B]),
 "0b_deep_type": ("run_r9_stage0b.py", "    if type(a) is not type(b):\n        return [f\"{path or '.'}: type", "    if False:\n        return [f\"{path or '.'}: type", [T0B]),
 "0b_rc_tau_fields": ("run_r9_stage0b.py", 'RC_TAU_FIELDS = ("taus", "percentiles", "n_values", "config")', 'RC_TAU_FIELDS = ("percentiles", "n_values", "config")', [T0B]),
 "0b_reader_excl_halves": ("run_r9_stage0b.py", '"banks_npz_sha256", "heads_files_sha256", "banks_files_sha256", "model_spec", "exploratory")', '"banks_npz_sha256", "heads_files_sha256", "banks_files_sha256", "model_spec", "exploratory", "halves")', [T0B]),
 "0b_tolerance": ("run_r9_stage0b.py", "TOL_R1_POINTS = 0.05 ", "TOL_R1_POINTS = 0.5 ", [T0B]),
 "0b_step1_rtol": ("run_r9_stage0b.py", "STEP1_RTOL = 1e-5 ", "STEP1_RTOL = 1e-2 ", [T0B]),
 # stage 2
 "2_made_differs": ("run_r9_stage2.py", "require(bool(diff), f'made {key} equals CLIP')  # guard:made_differs", "pass  # guard:made_differs", [T2]),
 "2_copied_equal": ("run_r9_stage2.py", "require(not diff, f'copied {key} differs from CLIP: {diff}')", "pass", [T2]),
 "2_prov_overlap": ("run_r9_stage2.py", "require(not set(made) & set(copied), 'made/copied keys overlap')", "pass", [T2]),
 "2_frozen_pins_sha": ("run_r9_stage2.py", "require(role in pins and sha == pins[role]['sha256'], f'frozen {role} differs from pins_clip.json')", "pass", [T2]),
 "2_frozen_r6_pin": ("run_r9_stage2.py", "require(sha == want, f'frozen {role} differs from its round 6 pin')", "pass", [T2]),
 "2_frozen_npz_arrays": ("run_r9_stage2.py", "            assert_array_provenance(a, b, [], keys)", "            pass", [T2]),
 "2_frozen_pm_rows": ("run_r9_stage2.py", "require(a['fit_rows'] == b['fit_rows'], 'frozen PM fit rows differ')", "pass", [T2]),
 "2_script_sha": ("run_r9_stage2.py", "require(want and path.is_file() and record.get('script_sha256') == want == R.sha256_file(path),", "require(True or want and path.is_file() and record.get('script_sha256') == want == R.sha256_file(path),", [T2]),
 "2_script_0b_passed": ("run_r9_stage2.py", "    require(item.get('passed') is True, 'stage 0b script checks have not passed')", "    pass", [T2]),
 "2_modules": ("run_r9_stage2.py", "require(stage0b.get('module_sha256', {}).get(module) == sha,", "require(True or stage0b.get('module_sha256', {}).get(module) == sha,", [T2]),
 "2_stage2_passed": ("run_r9_stage2.py", "require(stage.get('passed') is True and stage.get('backbone') == backbone,", "require(True or stage.get('passed') is True and stage.get('backbone') == backbone,", [T2]),
 "2_picks_passed": ("run_r9_stage2.py", "require(read_json(parts['picks_seed42']).get('passed') is True, 'picks file has not passed')", "pass", [T2]),
 "2_unrefit_nan": ("run_r9_stage2.py", "require(np.isnan(a[k]).all(), f'{role}/{k}: unused head must be NaN')", "pass", [T2]),
 "2_prov_keys_complete": ("run_r9_stage2.py", "require(set(a.files) == set(producer_made) | set(producer_copied) | set(unrefit),", "require(True or set(a.files) == set(producer_made) | set(producer_copied) | set(unrefit),", [T2]),
 "2_reader_feature_names": ("run_r9_stage2.py", "require(read_json(path)['feature_names'] == read_json(clip)['feature_names'], 'reader feature layout differs')", "pass", [T2]),
 "2_picked_ckpt": ("run_r9_stage2.py", "require(parts is not None and a['checkpoint_sha256'] == R.sha256_file(parts['A3_seed42']),", "require(True or parts is not None and a['checkpoint_sha256'] == R.sha256_file(parts['A3_seed42']),", [T2]),
 "2_targets_null": ("run_r9_stage2.py", "require('targets' in a and a['targets'] is None, 'new-backbone picks targets must be null')", "pass", [T2]),
 "2_input_not_backbone": ("run_r9_stage2.py", "                require(any(Path(parts[r]).resolve() == p.resolve() for r in recipe_roles),", "                require(True or any(Path(parts[r]).resolve() == p.resolve() for r in recipe_roles),", [T2]),
 "2_input_sha": ("run_r9_stage2.py", "require(any(q.is_file() and R._sha_cached(q) == value for q in candidates),", "require(True or any(q.is_file() and R._sha_cached(q) == value for q in candidates),", [T2]),
 "2_local_groups": ("run_r9_stage2.py", "require(sha == value, 'local_groups input array changed')", "pass", [T2]),
 "2_feature_store": ("run_r9_stage2.py", "require(value == stores[key], f'{role}: feature store hash differs: {key}')", "pass", [T2]),
 "2_features_complete": ("run_r9_stage2.py", "require(features == stores, f'{role}: feature-store provenance is incomplete')", "pass", [T2]),
 "2_record_backbone": ("run_r9_stage2.py", "require(rec.get('backbone', backbone) == backbone, f'{role}: record names another backbone')", "pass", [T2]),
 "2_record_file_sha": ("run_r9_stage2.py", "require(rec['file_sha256'] == sha, f'{role}: file SHA differs from its record')", "pass", [T2]),
 "2_wrong_producer": ("run_r9_stage2.py", "require(Path(rec.get('script', '')).name == expected_script, f'{role}: record names the wrong producer')", "pass", [T2]),
 "2_upstream": ("run_r9_stage2.py", "            require(upstream in parts and R._sha_cached(parts[upstream]) == want,", "            require(True or upstream in parts and R._sha_cached(parts[upstream]) == want,", [T2]),
 "2_ev_switched_off": ("run_r9_stage2.py", "require(not stage.get('switched_off'), 'not run: switched off; no pins of numbers')", "pass", [T2]),
 "2_ev_runner_rule": ("run_r9_stage2.py", "require(stage.get('runner_sha256') == R.sha256_file(__file__) and stage.get('rule_sha256') == R.RULE_SHA256,", "require(True or stage.get('runner_sha256') == R.sha256_file(__file__) and stage.get('rule_sha256') == R.RULE_SHA256,", [T2]),
 "2_ev_dry_run": ("run_r9_stage2.py", "require(stage['dry_run'] is dry_run, 'dry-run stage 2 cannot authorise a real backbone')", "pass", [T2]),
 "2_ev_stage0b_changed": ("run_r9_stage2.py", "require(path_record(stage0b_path) == stage['stage0b'], 'stage 0b changed')", "pass", [T2]),
 "2_ev_closure": ("run_r9_stage2.py", "require(stage['module_sha256'] == modules, 'stage 2 recipe module closure changed')", "pass", [T2]),
 "2_ev_receipt": ("run_r9_stage2.py", "require(path_record(receipt['path']) == receipt, f'{step}: receipt changed after stage 2')", "pass", [T2]),
 "2_ev_file_changed": ("run_r9_stage2.py", "require(path_record(paths[role]) == {k: p[k] for k in ('path', 'sha256')}, f'{role}: stage 2 file changed')", "pass", [T2]),
 "2_ev_record_changed": ("run_r9_stage2.py", "require(path_record(str(paths[role]) + '.record.json') == p['record'], f'{role}: stage 2 record changed')", "pass", [T2]),
 "2_ev_store_changed": ("run_r9_stage2.py", "require(R._sha_cached(file) == sha, f'feature store changed: {file}')", "pass", [T2]),
 "2_ev_frozen_changed": ("run_r9_stage2.py", "require(path_record(paths[role]) == stage['frozen_parts'][role], f'{role}: frozen file changed')", "pass", [T2]),
 "2_ev_seed42_changed": ("run_r9_stage2.py", "require(path_record(stage[name]['path']) == stage[name], f'{name}: changed after stage 2')", "pass", [T2]),
 "2_ev_frozen_assert": ("run_r9_stage2.py", "        assert_frozen(role, paths[role], BB.CLIP_PARTS[role])\n    for role in BB.RECIPE_ROLES:\n        assert_script", "        pass\n    for role in BB.RECIPE_ROLES:\n        assert_script", [T2]),
 "2_ev_script_assert": ("run_r9_stage2.py", "    for role in BB.RECIPE_ROLES:\n        assert_script(read_json(str(paths[role]) + '.record.json'), gate)", "    for role in BB.RECIPE_ROLES:\n        pass", [T2]),
 "2_sigma_finite": ("run_r9_stage2.py", "require(np.isfinite(value) and value >= 0, f'{c}: invalid {key}')", "pass", [T2]),
 "2_record_overwrite": ("run_r9_stage2.py", "        require(not path.exists(), f'refusing to overwrite {path}')", "        pass", [T2]),
 "2_switched_status": ("run_r9_stage2.py", "'status': 'not run: switched off' if cells['switched_off'] else 'ready',", "'status': 'ready',", [T2]),
 "2_switched_stop": ("run_r9_stage2.py", "    if cells['switched_off']:\n        write_new(stage_path, stage)", "    if False:\n        write_new(stage_path, stage)", [T2]),
 # pins
 "p_gate_parity": ("r9_pins.py", "g = np.asarray(gates[c])[tau][parity == 1 - h]", "g = np.asarray(gates[c])[tau][parity == h]", [TP, T2]),
 "p_gate_tau": ("r9_pins.py", "tau = int(cells[h]) // TAUS_PER_CELL", "tau = int(cells[h]) % 4", [TP, T2]),
 "p_gate_cond_b": ('r9_pins.py', 'for c in "ab":\n            g = np.asarray', 'for c in "a":\n            g = np.asarray', [TP, T2]),
 "p_switched_any": ("r9_pins.py", "return bool(all(F2.cell_values(int(c))[3] == 0 for c in cells))", "return bool(any(F2.cell_values(int(c))[3] == 0 for c in cells))", [TP, T2]),
 "p_switched_index": ("r9_pins.py", "return bool(all(F2.cell_values(int(c))[3] == 0 for c in cells))", "return bool(all(F2.cell_values(int(c))[2] == 0 for c in cells))", [TP, T2]),
 # r9_data joins
 "d_sample_id": ("r9_data.py", 'assert row["sample_id"] == sample_id, f"sample_id differs from CLIP: {row}"', 'pass', [TD]),
 "d_painting": ("r9_data.py", 'assert row["painting"] == painting, f"painting differs from CLIP: {row}"', 'pass', [TD]),
 "d_img_join": ("r9_data.py", 'assert index["images"][img_row]["painting"] == painting, f"image join painting differs: {row}"', 'pass', [TD]),
 "d_row_order_ids": ("r9_data.py", 'assert np.array_equal(data.sample_ids, reference.sample_ids), "sample_id row order differs from CLIP"', 'pass', [TD]),
 "d_row_order_paint": ("r9_data.py", 'assert np.array_equal(data.paintings, reference.paintings), "painting row order differs from CLIP"', 'pass', [TD]),
 "d_leak_group": ("r9_data.py", 'assert np.intersect1d(groups[left], groups[right]).size == 0, "a leakage group spans split parts"', 'pass', [TD]),
 "d_split_pin": ("r9_data.py", 'assert digest.hexdigest() == SPLIT_SHA256[role], f"{path}: SHA-256 differs from the CLIP pin"', 'pass', [TD]),
 "d_recomputed_split": ("r9_data.py", 'assert np.array_equal(mine, frozen), f"recomputed {name} differs from CLIP\'s pinned split"', 'pass', [TD]),
 "d_dup_painting": ("r9_data.py", 'assert len({row["painting"] for row in index["images"]}) == index["n_images"], "duplicate image painting"', 'pass', [TD]),
}

M.update({
 "0b_routed2": ("run_r9_stage0b.py", "        _require(Path(files[role]).resolve() != Path(PINNED[role]).resolve(),", "        _require(True,", [T0B, T2]),
 "0b_routed_missing": ("run_r9_stage0b.py", '_require(role in files, f"routed: no replay file for {role}")', 'pass', [T0B, T2]),
})
M.pop("0b_routed")
M.update({"g_graph_gpu": ("r9_recipe_groups.py", "if not cuda and not allow_cpu:   # guard:graph_gpu", "if False:   # guard:graph_gpu", ["test_r9_recipe_groups.py"])})

RC, TRC = "rederive/rd9_compare.py", ["rederive/rd9_test_compare.py", "rederive/rd9_test_arithmetic.py::test_phase1_agreement_checks_every_pinned_quantity"]
M.update({
 "c_switched_off": (RC, "for key in ('aff','cf','sigma_star','switched_off','gate_share'):", "for key in ('aff','cf','sigma_star','gate_share'):", TRC),
 "c_sigma": (RC, "for key in ('aff','cf','sigma_star','switched_off','gate_share'):", "for key in ('aff','cf','switched_off','gate_share'):", TRC),
 "c_cf": (RC, "for key in ('aff','cf','sigma_star','switched_off','gate_share'):", "for key in ('aff','sigma_star','switched_off','gate_share'):", TRC),
 "c_aff": (RC, "for key in ('aff','cf','sigma_star','switched_off','gate_share'):", "for key in ('cf','sigma_star','switched_off','gate_share'):", TRC),
 "c_b0": (RC, "    for key in ('B','B0'):\n        v=picks.get(key)", "    for key in ('B',):\n        v=picks.get(key)", TRC),
 "c_rca": (RC, "c.value('picks.rca_lambda',own['picks']['rca_lambda'],lambdas)", "pass", TRC),
 "c_taus": (RC, "c.value('picks.taus',own['picks']['taus'],tau.get('taus'))", "pass", TRC),
 "c_means": (RC, "c.value('seed42.mean_r1.'+name,got,seed42.get('mean_r1',{}).get(name))", "pass", TRC),
 "c_points": (RC, "c.value('seed42.checks.'+j+'.point',own['stats']['checks'][j]['point'],target.get('point'))", "pass", TRC),
 "c_nj": (RC, "c.value('seed42.checks.'+j+'.n',own['stats']['checks'][j]['n'],target.get('n_j',target.get('n')))", "pass", TRC),
 "c_hashes": (RC, "c.value('seed42.arrays.'+key,observed,entry)", "pass", TRC),
 "c_tau_tol": (RC, "return bool(np.isclose(ours,theirs,rtol=1e-9,atol=1e-15))", "return bool(np.isclose(ours,theirs,rtol=1e-3))", TRC),
 "c_picks_exact": (RC, "    if key.startswith('picks.') and 'taus' not in key:\n        return ours==theirs", "    if False:\n        return ours==theirs", TRC),
})

M.update({
 "2_replay_skip_widened": ("run_r9_stage2.py", "elif replay and value == file_hashes[role]:", "elif replay and value in by_hash:", [T2]),
})

TA, TP_, TD_ = ["test_r9_a28.py"], ["test_r9_a28_prepare.py"], ["test_r9_a28_data.py"]
M.update({
 "b1_min_paintings": ("r9_a28.py", "MIN_PAINTINGS = 30", "MIN_PAINTINGS = 29", TA),
 "b1_groups_rows": ("r9_a28.py", "groups = np.unique(kept.painting.to_numpy(), return_inverse=True)[1]", "groups = np.arange(len(kept))", TA),
 "b1_expected_rows": ("r9_a28.py", '"rows": 74_569', '"rows": 74_570', TA),
 "b1_expected_v_style": ("r9_a28.py", '"style": [9, 12, 17, 20, 21, 23, 24]', '"style": [9, 12, 17, 20, 21, 23]', TA),
 "b1_expected_v_sizes": ("r9_a28.py", 'EXPECTED_V_SIZES = {"emotion": 8, "style": 7, "genre": 8}', 'EXPECTED_V_SIZES = {"emotion": 8, "style": 6, "genre": 8}', TA),
 "b1_train_side_val": ("run_r9_a28_prepare.py", "np.union1d(split.train, split.val)", "split.train", TP_),
 "b1_prepare_counts": ("run_r9_a28_prepare.py", "assert got == A.EXPECTED,", "assert True,", TP_),
 "b1_prepare_v": ("run_r9_a28_prepare.py", "assert v == A.EXPECTED_V, v", "assert True, v", TP_),
 "b1_translation_painting": ("r9_a28.py", "(t.painting.to_numpy() == rows.painting.to_numpy()).all()", "True", TA),
 "b1_one_label": ("r9_a28.py", "assert (per == 1).all()", "assert True", TA),
 "b1_one_image": ("r9_a28_data.py", "len(set(image_name)) == len(image_name)", "True", TD_),
})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(mid, extra_k=None):
    fname, old, new, tests = M[mid]
    path = F / fname
    src = path.read_bytes(); before = sha(path)
    text = src.decode()
    n = text.count(old)
    if n != 1:
        return {"id": mid, "error": f"{n} matches"}
    path.write_text(text.replace(old, new))
    try:
        t = time.time()
        cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--deselect",
               "src/test/20261202_r9_scope/test_r9_pins.py::test_no_r9_module_imports_an_r6_module",
               *[str(F / x) for x in tests]]
        env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1",
               "CUDA_VISIBLE_DEVICES": ""}
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=150, env=env, cwd="/project/CoSiR-wt/r9-fx-tests")
        lines = p.stdout.strip().splitlines()
        failed = [l for l in lines if l.startswith("FAILED") or l.startswith("ERROR")]
        res = {"id": mid, "file": fname, "rc": p.returncode, "caught": p.returncode != 0,
               "summary": lines[-1] if lines else "", "first_failure": failed[0][:200] if failed else None,
               "seconds": round(time.time() - t, 1)}
    finally:
        path.write_bytes(src)
        assert sha(path) == before, f"restore failed for {path}"
    return res
if __name__ == "__main__":
    import signal
    def _term(*_):
        raise SystemExit("terminated")             # so run()'s finally restores the file
    signal.signal(signal.SIGTERM, _term)
    START, BUDGET = time.time(), float(os.environ.get("FR_BUDGET", "150"))
    out = Path(__file__).with_suffix(".json")
    allres = json.loads(out.read_text()) if out.exists() else {}
    ids = sys.argv[1:] if sys.argv[1:] != ["all"] else list(M)
    for mid in ids:
        if time.time() - START > BUDGET:
            print(mid, "not run: time budget", flush=True)
            continue
        r = run(mid)
        allres[mid] = r
        out.write_text(json.dumps(allres, indent=1) + "\n")
        print(mid, r.get("caught"), r.get("error", ""), r.get("summary", "")[:80], r.get("first_failure") or "", flush=True)
