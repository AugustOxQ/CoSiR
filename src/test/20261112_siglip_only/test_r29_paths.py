"""Tests of r29_paths' older-round import check: a mixed import (one older-round module from another checkout) fails."""
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
