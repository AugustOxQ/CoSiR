"""Tests of r29_paths' older-round import check: a mixed import (one older-round module from another checkout) fails."""
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402


def fake(monkeypatch, name, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, mod)


def test_older_rounds_point_at_main_checkout():
    for folder in (P.R28_ROUND, P.R14_ROUND, P.R23_ROUND, P.R17_ROUND, P.APC_ROUND):
        assert folder.parent == P.MAIN / 'src' / 'test'


def test_use_older_rounds_puts_main_folders_first():
    P.use_older_rounds()
    assert sys.path[:3] == [str(P.R28_ROUND), str(P.R17_ROUND), str(P.APC_ROUND)]


def test_check_passes_on_main_copy_and_own_round(monkeypatch):
    fake(monkeypatch, 'zz_fake_main', P.OLDER / '20261027_recipe_sweep' / 'zz_fake_main.py')
    fake(monkeypatch, 'zz_fake_own', P.ROUND / 'zz_fake_own.py')
    P.check_older_imports()


def test_mixed_import_fails(monkeypatch):
    other = Path('/project/CoSiR-r29-imp/src/test/20261027_recipe_sweep/zz_fake_rs.py')
    assert other.parent != P.OLDER / '20261027_recipe_sweep'
    fake(monkeypatch, 'zz_fake_rs', other)
    with pytest.raises(AssertionError, match='mixed import'):
        P.check_older_imports()


def test_mixed_import_fails_for_any_checkout(monkeypatch):
    fake(monkeypatch, 'zz_fake_other', Path('/somewhere/else/src/test/20261021_d2_ablation/zz_fake_other.py'))
    with pytest.raises(AssertionError, match='zz_fake_other'):
        P.check_older_imports()


# ------------------------------------------------------------------------------------------------ require_committed
def _repo(tmp_path):
    import subprocess
    repo = tmp_path / 'repo'
    (repo / 'records').mkdir(parents=True)

    def run(*a):
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *a], check=True, capture_output=True)
    run('init', '-q')
    (repo / 'README').write_text('x')           # a first commit, so HEAD exists
    run('add', 'README')
    run('commit', '-q', '-m', 'first')
    return repo, run


def test_require_committed_refuses_absent_untracked_staged_and_modified_and_accepts_committed(tmp_path):
    repo, run = _repo(tmp_path)
    f = repo / 'records' / 'pass_rule.json'
    with pytest.raises(RuntimeError, match='no pass_rule.json.*because'):
        P.require_committed(f, 'because')
    f.write_text('{"status": "fail"}\n')
    with pytest.raises(RuntimeError, match='not committed unchanged.*because'):
        P.require_committed(f, 'because')                         # untracked
    run('add', 'records')
    with pytest.raises(RuntimeError, match='not committed unchanged'):
        P.require_committed(f, 'because')                         # staged only
    run('commit', '-q', '-m', 'rule')
    P.require_committed(f, 'because')                             # committed unchanged (a "fail" status is fine: D9)
    f.write_text('{"status": "pass"}\n')
    with pytest.raises(RuntimeError, match='not committed unchanged'):
        P.require_committed(f, 'because')                         # modified after the commit


def test_require_committed_refuses_a_file_outside_any_repository(tmp_path):
    f = tmp_path / 'pass_rule.json'
    f.write_text('{}')
    with pytest.raises(RuntimeError, match='not committed'):
        P.require_committed(f, 'because')


# ---------------------------------------------------------------------------------------------- stage2_gate (D3 as amended by A1)
def _gate_repo(tmp_path, monkeypatch, rule=None, a1=True, commit=True, flag='A1'):
    """A throwaway git repo with records/pass_rule.json (a dict, or None for none) and records/amendment_A1.json (A1's shape)."""
    repo, run = _repo(tmp_path)
    rec = repo / 'records'
    rec.mkdir(exist_ok=True)
    names = []
    if rule is not None:
        (rec / 'pass_rule.json').write_text(json.dumps(rule))
        names.append('pass_rule.json')
    if a1:
        doc = dict(amendment='A1', applies_to=dict(pass_rule_status='fail', pass_rule_level=0, arm='siglip1_prefix'), note=P.A1_LINE)
        (rec / 'amendment_A1.json').write_text(json.dumps(doc))
        names.append('amendment_A1.json')
    if commit and names:
        run('add', '-f', *[f'records/{n}' for n in names])
        run('commit', '-q', '-m', 'records')
    if flag is None:
        monkeypatch.delenv(P.A1_ENV, raising=False)
    else:
        monkeypatch.setenv(P.A1_ENV, flag)
    return rec, run


FAIL = dict(status='fail', level=0, arm='siglip1_prefix')


def test_stage2_gate_pass_needs_no_flag_and_no_record(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, dict(status='pass', level=1, arm='x'), a1=False, flag=None)
    assert P.stage2_gate(rec) == 'pass'
    monkeypatch.setenv(P.A1_ENV, 'A1')
    assert P.stage2_gate(rec) == 'pass'                         # the flag on a pass changes nothing
    assert P.a1_marks('pass') == {} and P.a1_marks(None) == {}


def test_stage2_gate_fail_without_the_flag_refuses_even_with_a_committed_record(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, FAIL, flag=None)
    with pytest.raises(RuntimeError, match='--amendment A1'):
        P.stage2_gate(rec)


def test_stage2_gate_fail_with_the_flag_and_a_committed_matching_record_gives_a1(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, FAIL)
    assert P.stage2_gate(rec) == 'A1'
    assert P.a1_marks('A1') == dict(amendment='A1', amendment_line='stage 2 ran on the fail under A1')


def test_stage2_gate_refuses_another_flag_value(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, FAIL, flag='A2')
    with pytest.raises(RuntimeError, match='only "A1"'):
        P.stage2_gate(rec)


def test_stage2_gate_flag_with_the_record_missing_refuses(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, FAIL, a1=False)
    with pytest.raises(RuntimeError, match='no amendment_A1.json'):
        P.stage2_gate(rec)


def test_stage2_gate_flag_with_the_record_untracked_or_edited_refuses(tmp_path, monkeypatch):
    rec, run = _gate_repo(tmp_path, monkeypatch, FAIL, a1=True, commit=False)
    run('add', '-f', 'records/pass_rule.json')
    run('commit', '-q', '-m', 'rule only')
    with pytest.raises(RuntimeError, match='amendment_A1.json is not committed'):          # untracked
        P.stage2_gate(rec)
    run('add', '-f', 'records/amendment_A1.json')
    run('commit', '-q', '-m', 'a1')
    assert P.stage2_gate(rec) == 'A1'
    doc = json.loads((rec / 'amendment_A1.json').read_text())
    (rec / 'amendment_A1.json').write_text(json.dumps(dict(doc, extra=1)))
    with pytest.raises(RuntimeError, match='amendment_A1.json is not committed'):          # edited after the commit
        P.stage2_gate(rec)


def test_stage2_gate_flag_with_an_uncommitted_pass_rule_refuses(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, FAIL)
    (rec / 'pass_rule.json').write_text(json.dumps(dict(FAIL, extra=1)))
    with pytest.raises(RuntimeError, match='pass_rule.json is not committed'):
        P.stage2_gate(rec)


@pytest.mark.parametrize('rule,a1_change', [
    (dict(FAIL, level=1), {}),                                                  # level differs
    (dict(FAIL, arm='siglip1'), {}),                                            # arm differs
    (FAIL, dict(amendment='A2')),                                               # not A1's record
    (FAIL, dict(note='something else')),
    (FAIL, dict(applies_to=dict(pass_rule_status='pass', pass_rule_level=0, arm='siglip1_prefix')))])  # status differs
def test_stage2_gate_flag_with_a_record_that_does_not_match_refuses(tmp_path, monkeypatch, rule, a1_change):
    rec, run = _gate_repo(tmp_path, monkeypatch, rule)
    if a1_change:
        doc = json.loads((rec / 'amendment_A1.json').read_text())
        (rec / 'amendment_A1.json').write_text(json.dumps(dict(doc, **a1_change)))
        run('add', '-f', 'records/amendment_A1.json')
        run('commit', '-q', '-m', 'edit a1')
    with pytest.raises(RuntimeError, match='not the record of A1|does not match'):
        P.stage2_gate(rec)


@pytest.mark.parametrize('rule', [dict(status='parked', level=0, arm='x'), dict(level=0, arm='x'), dict(status=None)])
def test_stage2_gate_parked_or_statusless_refuses_even_with_the_flag(tmp_path, monkeypatch, rule):
    rec, _ = _gate_repo(tmp_path, monkeypatch, rule)
    with pytest.raises(RuntimeError, match='not "pass"'):
        P.stage2_gate(rec)


def test_stage2_gate_without_a_pass_rule_refuses(tmp_path, monkeypatch):
    rec, _ = _gate_repo(tmp_path, monkeypatch, None)
    with pytest.raises(RuntimeError, match='no .*pass_rule.json'):
        P.stage2_gate(rec)
