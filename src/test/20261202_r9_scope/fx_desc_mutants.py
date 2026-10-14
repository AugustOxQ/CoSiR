"""fx-desc: break each new guard once in place, run test_r9_descriptive.py, restore the file byte for byte.

python fx_desc_mutants.py   (from the round folder; about 5 s per mutant). Writes fx_desc_mutation_results.json.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
RUN, DESC = HERE / 'run_r9_descriptive.py', HERE / 'r9_descriptive.py'

M = [  # name, file, old, new
    ('verdict_backbone_code_name', RUN, "RH.LEDGER_NAMES[RH.READ_BACKBONES[read]]", "RH.READ_BACKBONES[read]"),
    ('verdict_read_not_checked', RUN, "and verdict.get('read') == read\n", "\n"),
    ('verdict_status_not_checked', RUN, "verdict.get('status') == 'run' and verdict.get('verdict') in allowed", "verdict.get('verdict') in allowed or True"),
    ('verdict_rule_sha_not_checked', RUN, "verdict.get('rule_sha256') == R.RULE_SHA256 and", "True and"),
    ('pass_sha_binding_159', RUN, "refuse_unless(R.sha256_file(ppath) == verdict.get('held_pass_sha256') == agr['held_pass_sha256'], 'pass SHA binding')", "pass"),
    ('agreement_sha_in_search', RUN, "if p.is_file() and R.sha256_file(p) == verdict.get('agreement_sha256')", "if p.is_file()"),
    ('agreement_missing_not_refused', RUN, "refuse_unless(apath is not None, 'agreement missing or SHA mismatch')", "pass"),
    ('rebuilt_arrays_191', DESC, "        require(value.shape == want.shape and value.dtype == want.dtype and np.array_equal(value, want),\n                'rebuilt primary/raw array: ' + k)", "        pass"),
    ('rebuilt_dtype_not_compared', DESC, "value.dtype == want.dtype and ", ""),
    ('directions_172', DESC, "    require(all(np.array_equal(sc[s][m], .5 * (directions[s]['i2t'][m] + directions[s]['t2i'][m]))\n                for s in S.ALL_SCORERS for m in ROW_METRICS), 'direction averages reproduce core')", "    pass"),
    ('directions_any_not_all', DESC, "    require(all(np.array_equal(sc[s][m], .5", "    require(any(np.array_equal(sc[s][m], .5"),
    ('near_source_sha_139', RUN, "sha == rec['near_duplicates_source']['sha256'] == NEAR_DUPLICATES_SHA256", "sha == NEAR_DUPLICATES_SHA256"),
    ('near_pin_139', RUN, "sha == rec['near_duplicates_source']['sha256'] == NEAR_DUPLICATES_SHA256", "sha == rec['near_duplicates_source']['sha256']"),
    ('near_cos_092_order', RUN, "<= near['cos_098'] <= near['cos_092'] <= 746", "<= near['cos_098'] <= 746"),
    ('near_cos_092_int', RUN, "for k in ('cos_099', 'cos_098', 'cos_092'))\n            and 0 <=", "for k in ('cos_099', 'cos_098'))\n            and 0 <="),
    ('near_counts_vs_pass', RUN, "and {k: near[k] for k in ('cos_099', 'cos_098')} == rec['near_duplicates'], 'near-duplicate counts')", ", 'near-duplicate counts')"),
    ('stage0b_rule_145', RUN, "rec.get('rule_sha256') == R.RULE_SHA256 and rec.get('scored') is True", "rec.get('scored') is True"),
    ('stage0b_scored_145', RUN, "rec.get('rule_sha256') == R.RULE_SHA256 and rec.get('scored') is True", "rec.get('rule_sha256') == R.RULE_SHA256"),
    ('smoke_tag_plain_name', RUN, "(smoke and RH.SUBDIR_RE.fullmatch(smoke_tag))", "(smoke)"),
    ('smoke_tag_needs_smoke', RUN, "(smoke and RH.SUBDIR_RE.fullmatch(smoke_tag))", "(RH.SUBDIR_RE.fullmatch(smoke_tag))"),
    ('output_exists', RUN, "refuse_unless(not target.exists(), 'output exists: ' + str(target))", "pass"),
]
out = {}
for name, path, old, new in M:
    raw = path.read_bytes()
    text = raw.decode()
    assert text.count(old) == 1, (name, text.count(old))
    path.write_text(text.replace(old, new))
    try:
        r = subprocess.run([PY, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', 'test_r9_descriptive.py'],
                           cwd=HERE, capture_output=True, text=True, timeout=200)
    finally:
        path.write_bytes(raw)
    failed = [l for l in r.stdout.splitlines() if l.startswith('FAILED')]
    out[name] = {'caught': r.returncode != 0, 'first_failure': failed[:1]}
    print(name, out[name]['caught'], failed[:1], flush=True)
(HERE / 'fx_desc_mutation_results.json').write_text(json.dumps(out, indent=1) + '\n')
