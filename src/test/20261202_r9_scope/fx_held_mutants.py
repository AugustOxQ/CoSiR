"""fx-held: temporary source mutants of the launch gate, the held runner and the apply step; each restored in finally.

Usage: fx_held_mutants.py <first> <last> <out.json>. Each mutant runs the test files named with it;
caught = pytest exit != 0. The source file's SHA-256 is checked equal to the original after every mutant.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

F = Path(__file__).resolve().parent
PY = '/root/miniconda3/envs/CoSiR/bin/python'
P, H, A = 'run_r9_preread.py', 'run_r9_held.py', 'run_r9_apply_rule.py'
TP, TA, TH = 'test_r9_preread.py', 'test_r9_apply_rule.py', 'test_r9_held.py'
MUT = [
    # C MEDIUM-1: the launch gate's parts
    ('launch module_sha256 half', P, "require(rec.get('module_sha256') == R.r9_module_shas(here)\n            and rec.get('runner_sha256')", "require(rec.get('runner_sha256')", TP),
    ('launch read field', P, "and rec.get('read') == read\n", "\n", TP),
    ('launch backbone field', P, "and rec.get('backbone') == RH.READ_BACKBONES[read]\n", "\n", TP),
    ('launch all checks passed', P, "and all(rec.get('checks', {}).get(k, {}).get('status') == 'passed' for k in CHECK_ORDER),", ",", TP),
    ('launch non-empty inputs', P, "require(bool(rec.get('input_file_sha256'))\n            and all(", "require(all(", TP),
    # C LOW-4
    ('launch files: whole check', P, "    require(all(str(Path(p).resolve()) in rec['input_file_sha256'] for p in own),", "    require(True,", TP),
    ('launch files: pins dropped', P, "own = [R.pins_path(RH.READ_BACKBONES[read]), R.RESULTS / S0A.OUT_NAME]", "own = [R.RESULTS / S0A.OUT_NAME]", TP),
    ('launch files: stage 0a dropped', P, "own = [R.pins_path(RH.READ_BACKBONES[read]), R.RESULTS / S0A.OUT_NAME]", "own = [R.pins_path(RH.READ_BACKBONES[read])]", TP),
    ('launch files: A28 pins dropped', P, " + ([RH.AD.PINS_PATH] if read in RH.A28_READS else [])", "", TP),
    # C MEDIUM-2
    ('stage2 switched_off', P, "and stage.get('switched_off') is False and stage.get('dry_run') is False", "and stage.get('dry_run') is False", TP),
    ('stage2 dry_run', P, " and stage.get('dry_run') is False,", ",", TP),
    ('stage2 backbone', P, "stage.get('passed') is True and stage.get('backbone') == backbone", "stage.get('passed') is True", TP),
    ('regression fields seed only', P, "fields = ('seed', 'mean_r1', 'checks', 'arrays', 'episodes_sha256', 'coef_sha256')", "fields = ('seed',)", TP),
    ('regression field mean_r1 dropped', P, "fields = ('seed', 'mean_r1', 'checks',", "fields = ('seed', 'checks',", TP),
    ('regression field coef_sha256 dropped', P, "'episodes_sha256', 'coef_sha256')\n        require(all(", "'episodes_sha256')\n        require(all(", TP),
    ('clip regression items only', P, "require(len(items) == S0A.N_ITEMS and all(v['equal'] is True for v in items.values())\n                and not S0A.compare_counts(counts, refs[S0A.COUNTS_NAME]),", "require(len(items) == S0A.N_ITEMS and all(v['equal'] is True for v in items.values()),", TP),
    ('clip regression counts only', P, "\n                and not S0A.compare_counts(counts, refs[S0A.COUNTS_NAME]),", ",", TP),
    ('clip regression item count dropped', P, "require(len(items) == S0A.N_ITEMS and all(v['equal']", "require(all(v['equal']", TP),
    ('phase1 phase==1', P, "and agr.get('phase') == 1 ", "", TP),
    ('phase1 backbone', P, "and agr.get('backbone') == state.backbone\n", "\n", TP),
    ('phase1 items equal', P, "\n            and all(v.get('equal') is True for v in items),", ",", TP),
    ('phase1 failed==0', P, "and agr.get('failed') == 0 ", "", TP),
    ('phase1 non-empty items', P, "and bool(items) and agr.get('compared')", "and agr.get('compared')", TP),
    ('phase1 compared==len', P, " and agr.get('compared') == len(items)", "", TP),
    # C LOW-1
    ('main: no setdefault', H, "    sys.modules.setdefault('run_r9_held', sys.modules[__name__])\n", "", TP),
    # C LOW-2
    ('heldfix1 pre-read name', P, "preread_name(read, 'heldfix1' if variant == 'fix1' else variant)", "preread_name(read, variant)", TP),
    ('heldfix1 cli kind', P, "'heldfix1' if a.held_fix else ", "", TP),
    ('heldfix1 smoke: held read_names', H, '"heldfix1" if variant == "fix1" else "reserve" if variant == "reserve"', '"fix1" if variant == "fix1" else "reserve" if variant == "reserve"', TP + ' ' + TH),
    ('heldfix1 smoke: held latest', H, 'SMOKE_RECORDS["heldfix1" if variant == "fix1" else variant]', 'SMOKE_RECORDS[variant]', TP + ' ' + TH),
    ('heldfix1 smoke: held precedence', H, 'for kind in ("heldfix1", "fix3", "fix2", "fix1", "crash1"):', 'for kind in ("fix3", "fix2", "fix1", "crash1"):', TP),
    ('heldfix1 smoke: apply precedence', A, 'for kind in ("heldfix1", "fix3", "fix2", "fix1", "crash1"):', 'for kind in ("fix3", "fix2", "fix1", "crash1"):', TP),
    # C LOW-3
    ('smoke agreement also in rederive dir', P, "    write_once(copy, agr)\n", "    write_once(AR.AGREEMENT_DIR / 'smoke' / tag / copy.name, agr)\n    write_once(copy, agr)\n", TP),
    ('scratch smoke record copy always', P, "targets = (path, HERE / path.name) if Path(results).resolve() == Path(R.RESULTS).resolve() else (path,)", "targets = (path, HERE / path.name)", TP),
    # A1
    ('apply: L1 print', A, "f\"re-derivation's count {b['rederivation_n']}, largest", "f\"largest", TA),
    ('apply: L2 quantity', A, 'c.get("quantity") == ST.QUANTITIES[name][2]', 'isinstance(c.get("quantity"), str)', TA),
    ('apply: L3 read', A, 'rec.get("read") != read or rec.get("backbone") != READ_BACKBONE_IDS[read]', 'rec.get("backbone") != READ_BACKBONE_IDS[read]', TA),
    ('apply: L3 backbone', A, 'rec.get("read") != read or rec.get("backbone") != READ_BACKBONE_IDS[read]', 'rec.get("read") != read', TA),
    ('held: L3 pass-file fields', H, 'pr.update({"read": spec.read, "backbone": backbone, "kind"', 'pr.update({"kind"', TH),
    ('assert_rule: rule sha', A, "    if CM.sha256_bytes(raw) != CM.RULE_SHA256:\n", "    if False:\n", TA),
    ('assert_rule: r6 sha', A, "    if CM.sha256_file(r6) != CM.R6_RULE_SHA256:\n", "    if False:\n", TA),
    ('assert_rule: rule phrases', A, "    missing = [p for p in RULE_PHRASES if p not in text]\n", "    missing = []\n", TA),
    ('assert_rule: r6 phrases', A, "    missing += [p for p in R6_PHRASES if p not in r6_text]\n", "", TA),
    ('assert_rule: call in not-run path', A, "    if not_run is not None:\n        assert_rule()\n", "    if not_run is not None:\n", TA),
    ('assert_rule: call in read path', A, "    smoke_record = None if smoke else smoke_guard(out, variant)\n    assert_rule()\n", "    smoke_record = None if smoke else smoke_guard(out, variant)\n", TA),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a, b, out):
    res = []
    for i, (name, fname, old, new, tests) in enumerate(MUT):
        if not a <= i <= b:
            continue
        path = F / fname
        orig = path.read_bytes()
        before = hashlib.sha256(orig).hexdigest()
        text = orig.decode()
        if text.count(old) != 1:
            res.append({'i': i, 'name': name, 'status': f'ANCHOR x{text.count(old)}'})
            continue
        try:
            path.write_text(text.replace(old, new))
            r = subprocess.run([PY, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', *tests.split()], cwd=F,
                               capture_output=True, text=True,
                               env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
            first = next((l for l in r.stdout.splitlines() if l.startswith('FAILED')), '')
            res.append({'i': i, 'name': name, 'caught': r.returncode != 0, 'first_failure': first[:140]})
        finally:
            path.write_bytes(orig)
        assert sha(path) == before, f'{fname} not restored'
        print(i, name, res[-1].get('caught'), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
