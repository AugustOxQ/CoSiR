"""Final review B1: temporary guard mutations, each reverted at once (try/finally, SHA checked after restore).

For each mutant: replace one exact text (the n-th occurrence) in one module, run the named test files (one pytest
process per file, -x), record whether any test failed (caught) or all passed (survived), restore the file.
Writes fr_b1_mutations.json beside this file.  Usage: python fr_b1_mutations.py [name ...]
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
ROOT = F.parents[2]
PY = "/root/miniconda3/envs/CoSiR/bin/python"
ENV = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1",
       "CUDA_VISIBLE_DEVICES": ""}
DATA_T = ["test_r9_a28_data.py"]
A28_T = ["test_r9_a28.py"]
T = True
M = {
    # r9_a28_data.load_a28 joins and counts
    "row_id_join": ("r9_a28_data.py", "np.array_equal(store_row_id, row_id[kept_pos])", "True", 0, DATA_T),
    "caption_painting_join": ("r9_a28_data.py", "np.array_equal(store_painting, painting[kept_pos])", "True", 0, DATA_T),
    "one_image_per_painting": ("r9_a28_data.py", "len(set(image_name)) == len(image_name)", "True", 0, DATA_T),
    "store_paintings_eq_read": ("r9_a28_data.py", "set(image_name) == set(painting.tolist())", "True", 0, DATA_T),
    "img_pointer_range": ("r9_a28_data.py", "((0 <= point) & (point < len(images))).all()", "True", 0, DATA_T),
    "img_pointer_painting": ("r9_a28_data.py",
                             "np.array_equal(np.asarray(image_name, dtype=str)[point], store_painting)", "True", 0,
                             DATA_T),
    "texts_eq_kept_count": ("r9_a28_data.py", "len(texts) == len(kept_pos)", "True", 0, DATA_T),
    "stores_finite": ("r9_a28_data.py", "np.isfinite(img_store).all() and np.isfinite(txt_store).all()", "True", 0,
                      DATA_T),
    "img_shape": ("r9_a28_data.py", 'img_store.shape == (len(images), dim) == (index["n_images"], dim)', "True", 0,
                  DATA_T),
    "txt_shape": ("r9_a28_data.py", 'txt_store.shape == (len(texts), dim) == (index["n_texts"], dim)', "True", 0,
                  DATA_T),
    "index_header": ("r9_a28_data.py", '("artelingo28", backbone, False, "float16", dim)',
                     '(index["dataset"], index["backbone"], index["smoke"], index["dtype"], index["dim"])', 0, DATA_T),
    "float16": ("r9_a28_data.py", "img_store.dtype == txt_store.dtype == np.float16", "True", 0, DATA_T),
    "space_kept_finite": ("r9_a28_data.py", "finite[a.kept].all()", "True", 0, DATA_T),
    "space_dropped_nan": ("r9_a28_data.py", "not finite[a.dropped].any()", "True", 0, DATA_T),
    "space_kept_is_not_dropped": ("r9_a28_data.py", "(a.kept == ~a.dropped).all()", "True", 0, DATA_T),
    "space_lengths": ("r9_a28_data.py", "len(getattr(a, name)) == n", "True", 0, DATA_T),
    "table_row_order": ("r9_a28_data.py", "(np.diff(row_id) > 0).all()", "True", 0, DATA_T),
    "table_columns": ("r9_a28_data.py", "list(tab.columns) == list(TABLE_COLUMNS)", "True", 0, DATA_T),
    "table_languages": ("r9_a28_data.py", "languages == sorted(expected_languages)", "True", 0, DATA_T),
    "language_index": ("r9_a28_data.py", "(np.asarray(languages)[language - 1] == lang).all()", "True", 0, DATA_T),
    "loader_backbone_pinned": ("r9_a28_data.py", 'f"store/{backbone}/index.json" in files', "True", 0, DATA_T),
    # pins
    "pinned_sha_in_loader": ("r9_a28_data.py", "got == sha,", "True,", 0, DATA_T),
    "pinned_missing_in_loader": ("r9_a28_data.py", "_require(p.is_file(),", "_require(True,", 0, DATA_T),
    "check_pins_sha": ("r9_a28_data.py", "got[key] == sha,", "True,", 0, DATA_T),
    "check_pins_missing_file": ("r9_a28_data.py", "_require(p.is_file(),", "_require(True,", 1, DATA_T),
    "check_pins_absent_keys": ("r9_a28_data.py", "_require(not absent,", "_require(True,", 0, DATA_T),
    "check_pins_uncovered": ("r9_a28_data.py", "_require(not uncovered,", "_require(True,", 0, DATA_T),
    "check_pins_sha_format": ("r9_a28_data.py",
                              'isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)',
                              "True", 0, DATA_T),
    "check_pins_needs_results": ("r9_a28_data.py", 'need = {f"results/{f}" for f in RESULT_FILES}', "need = set()", 0,
                                 DATA_T),
    "check_pins_needs_index": ("r9_a28_data.py", 'STORE_FILES = ("img.npy", "txt.npy", "index.json")',
                               'STORE_FILES = ("img.npy", "txt.npy")', 0, DATA_T),
    "write_pins_no_overwrite": ("r9_a28_data.py", "_require(not out.exists(),", "_require(True,", 0, DATA_T),
    "write_pins_entry_missing": ("r9_a28_data.py", "_require(path.is_file(),", "_require(True,", 0, DATA_T),
    "merged_from": ("r9_a28_data.py", "_require(listed == want,", "_require(True,", 0, DATA_T),
    "held_runner_a28_pins": ("run_r9_held.py", "AD.check_pins(AD.PINS_PATH, backbones=(backbone,))  # guard:a28_pins",
                             "pass  # guard:a28_pins", 0, ["test_r9_held_a28.py"]),
    # r9_a28 build_read_set, flags, V
    "both_held_and_train": ("r9_a28.py", "assert not both,", "assert True,", 0, A28_T),
    "translation_cover": ("r9_a28.py",
                          "t.csv_row.is_unique and len(t) == len(rows) and np.array_equal(t.csv_row.to_numpy(), row_id)",
                          "True", 0, A28_T),
    "translation_painting": ("r9_a28.py", "(t.painting.to_numpy() == rows.painting.to_numpy()).all()", "True", 0, A28_T),
    "translation_language": ("r9_a28.py", "(t.language.to_numpy() == rows.language.to_numpy()).all()", "True", 0, A28_T),
    "one_label_per_painting": ("r9_a28.py", "assert (per == 1).all()", "assert True", 0, A28_T),
    "genre_eq_csv": ("r9_a28.py", "assert not bad,", "assert True,", 0, A28_T),
    "other_is_unknown": ("r9_a28.py", 'EMOTION_UNKNOWN = ("something else", "other")',
                         'EMOTION_UNKNOWN = ("something else",)', 0, A28_T),
    "dropped_strip": ("r9_a28.py", 'x.strip() == ""', 'x == ""', 0, A28_T),
    "failed_strict": ("r9_a28.py", "failed * 20 > rows", "failed * 20 >= rows", 0, A28_T),
    "script_strict": ("r9_a28.py", 'c["outputs_still_in_source_script"] * 20 > rows',
                      'c["outputs_still_in_source_script"] * 20 >= rows', 0, A28_T),
    "script_latin_only": ("r9_a28.py", "script != LATIN and ", "", 0, A28_T),
    "length_strict_low": ("r9_a28.py", "ratio < LENGTH_BAND[0]", "ratio <= LENGTH_BAND[0]", 0, A28_T),
    "length_strict_high": ("r9_a28.py", "ratio > LENGTH_BAND[1]", "ratio >= LENGTH_BAND[1]", 0, A28_T),
    "median_after_drops": ("r9_a28.py", "words[(lang == k) & ~dropped]", "words[(lang == k)]", 0, A28_T),
    "english_on_read_only": ("r9_a28.py", "eng = csv[on_read & english]", "eng = csv[english]", 0, A28_T),
    "v_kept_only": ("r9_a28.py", "kept = read_df[~read_df.dropped.to_numpy()]", "kept = read_df", 0, A28_T),
    "v_all_known": ("r9_a28.py", "known &= kept[a].to_numpy() >= 0", "known &= kept[a].to_numpy() >= -1", 0, A28_T),
    "v_inside_dev": ("r9_a28.py", "assert not outside,", "assert True,", 0, A28_T),
    "v_min_paintings": ("r9_a28.py", "MIN_PAINTINGS = 30", "MIN_PAINTINGS = 29", 0, A28_T),
    "v_groups_paintings": ("r9_a28.py", "groups = np.unique(kept.painting.to_numpy(), return_inverse=True)[1]",
                           "groups = np.arange(len(kept))", 0, A28_T),
    "expected_rows": ("r9_a28.py", '"rows": 74_569', '"rows": 74_570', 0, A28_T),
    "expected_v_style": ("r9_a28.py", '"style": [9, 12, 17, 20, 21, 23, 24]', '"style": [9, 12, 17, 20, 21, 23]', 0,
                         A28_T),
    "train_side_includes_val": ("run_r9_a28_prepare.py", "np.union1d(split.train, split.val)", "split.train", 0,
                                A28_T),
    "prepare_counts_assert": ("run_r9_a28_prepare.py", "assert got == A.EXPECTED,", "assert True,", 0, A28_T),
    "prepare_v_assert": ("run_r9_a28_prepare.py", "assert v == A.EXPECTED_V, v", "assert True, v", 0, A28_T),
}
RC_T = ["rederive/rd9_test_compare.py"]
M.update({
    "rc_flagged_compare": ("rederive/rd9_compare.py", "c.value('flagged_languages',", "(lambda *a: None)(", 0, RC_T),
    "rc_near_compare": ("rederive/rd9_compare.py", "c.value('near_duplicates.'+key,a,b)", "pass", 0, RC_T),
    "rc_near_only_099": ("rederive/rd9_compare.py", "for key in ('cos_099','cos_098'):", "for key in ('cos_099',):", 0,
                         RC_T),
    "rc_set_aside_compare": ("rederive/rd9_compare.py",
                             "c.value('set_aside',derived_set_aside(own),reported_set_aside(report.get('set_aside',MISSING)))",
                             "pass", 0, RC_T),
    "rc_set_aside_first_seed": ("rederive/rd9_compare.py", "for seed in own['stats']['seeds']:",
                                "for seed in own['stats']['seeds'][:1]:", 0, RC_T),
    "rc_flags_unpinned": ("rederive/rd9_compare.py",
                          "flags_raw=read_pinned(args.language_flags,pin_entries(json.loads(args.a28_pins.read_text())))",
                          "flags_raw=args.language_flags.read_bytes()", 0, RC_T),
    "rc_only_h8": ("rederive/rd9_compare.py", "own.get('read') in ('H8','H9','H10')", "own.get('read') in ('H8',)", 0,
                   RC_T),
    "rc_train_side_912": ("rederive/rd9_compare.py", "duplicates.get('train_side')==912", "True", 0, RC_T),
    "rc_read_paintings_746": ("rederive/rd9_compare.py", "duplicates.get('read_paintings')==746", "True", 0, RC_T),
    "rc_reported_set_aside_dup": ("rederive/rd9_compare.py", "assert len(entries)==len(set(entries))", "assert True",
                                  0, RC_T),
})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(name):
    rel, old, new, nth, tests = M[name]
    path = F / rel
    text = path.read_text()
    before = sha(path)
    parts = text.split(old)
    if len(parts) < nth + 2:
        return {"error": f"text found {len(parts) - 1} times"}
    mutated = old.join(parts[:nth + 1]) + new + old.join(parts[nth + 1:])
    res = {}
    try:
        path.write_text(mutated)
        for t in tests:
            p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(F / t)], cwd=ROOT,
                               env=ENV, capture_output=True, text=True, timeout=200)
            last = (p.stdout.strip().splitlines() or [""])[-1]
            res[t] = {"rc": p.returncode, "tail": last[-160:]}
    finally:
        path.write_text(text)
    assert sha(path) == before, f"{rel} not restored"
    res["caught"] = any(v["rc"] != 0 for k, v in res.items() if isinstance(v, dict))
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or list(M)
    out_path = HERE / "fr_b1_mutations.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    for n in names:
        done[n] = run(n)
        print(n, "caught" if done[n].get("caught") else ("ERROR " + done[n]["error"] if "error" in done[n] else
                                                         "SURVIVED"), flush=True)
        out_path.write_text(json.dumps(done, indent=1))
