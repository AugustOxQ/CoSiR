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
