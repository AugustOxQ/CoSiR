"""Final review C: the held launch gate against bad pre-read and smoke records (tmp_path files only).

Each scenario calls the real run_r9_held.run_held with the real preread_guard and smoke_guard; only the
ledger step (refuse_or_go), h5_hashes, a28_reference and run_read are stubbed. A launch = run_read called.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
from test_r9_preread import case, launch_record, put, U  # noqa: F401  (fixture)

RH = U.RH


def arm(c, read='H8'):
    launched = []
    names = RH.read_names(read, 'held')
    c.mp.setattr(RH, 'refuse_or_go', lambda *a: NS(names=names, attempts=[], want={}, ledger={}))
    c.mp.setattr(RH, 'h5_hashes', lambda *a: {})
    c.mp.setattr(RH, 'a28_reference', lambda *a: None)
    c.mp.setattr(RH, 'run_read', lambda *a: launched.append(read) or 0)
    return launched


def good_smoke(c, sha=None):
    put(c.root / U.AR.SMOKE_RECORDS['smoke'], {'passed': True, 'module_sha256': sha or U.R.r9_module_shas()})


def go(c, read='H8', **kw):
    launched = arm(c, read)
    code = RH.run_held(RH.READ_BACKBONES[read], read, results=c.root, **kw)
    return code, launched


def test_control_valid_record_launches(case):
    put(case.root / U.preread_name('H8'), launch_record(case)); good_smoke(case)
    code, launched = go(case)
    assert code == 0 and launched == ['H8']


def test_missing(case):
    good_smoke(case)
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_failed(case):
    rec = launch_record(case); rec['passed'] = False
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_check_not_run_but_passed_true(case):
    rec = launch_record(case); rec['checks']['smoke'] = {'status': 'not_run'}
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_stale_module_bytes(case):
    rec = launch_record(case); rec['module_sha256'] = {'recipe.py': 'older'}
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_other_read_same_backbone(case):
    # H6's passing record placed under H9's name (both Qwen)
    rec = launch_record(case); rec.update(read='H6', backbone='qwen3vl_emb_2b')
    put(case.root / U.preread_name('H9'), rec); good_smoke(case)
    assert go(case, 'H9') == (RH.EXIT_REFUSE, [])


def test_other_backbone(case):
    rec = launch_record(case); rec.update(read='H9', backbone='clip_b32')
    put(case.root / U.preread_name('H9'), rec); good_smoke(case)
    assert go(case, 'H9') == (RH.EXIT_REFUSE, [])


def test_smoke_from_other_bytes(case):
    put(case.root / U.preread_name('H8'), launch_record(case)); good_smoke(case, {'recipe.py': 'older'})
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_smoke_missing(case):
    put(case.root / U.preread_name('H8'), launch_record(case))
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_empty_inputs(case):
    rec = launch_record(case); rec['input_file_sha256'] = {}
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (RH.EXIT_REFUSE, [])


def test_reserve_needs_reserve_preread(case):
    put(case.root / U.preread_name('H8'), launch_record(case))
    put(case.root / U.AR.SMOKE_RECORDS['reserve'], {'passed': True, 'module_sha256': U.R.r9_module_shas()})
    assert go(case, reserve=True) == (RH.EXIT_REFUSE, [])


def test_fix_needs_fix_preread(case):
    put(case.root / U.preread_name('H8'), launch_record(case))
    put(case.root / U.AR.SMOKE_RECORDS['fix1'], {'passed': True, 'module_sha256': U.R.r9_module_shas()})
    assert go(case, fix=1) == (RH.EXIT_REFUSE, [])


def test_malformed_record_does_not_launch(case):
    (case.root / U.preread_name('H8')).write_text('[1, 2]'); good_smoke(case)
    try:
        code, launched = go(case)
    except Exception:
        launched = []
    assert launched == []
