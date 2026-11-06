"""D13 wait rule of the runner (ticket 09). The rest of the runner is proved by the smoke run."""
from datetime import datetime

import json
import pytest

import tcl_data
import tcl_paths as P
import tcl_run as R
from tcl_run import heavy_wait_seconds as wait

H = 3600


def test_weekday_before_ten_and_fits():
    assert wait(datetime(2026, 11, 9, 8, 0), 60) == 0                  # Monday 08:00 to 09:00


def test_weekday_before_ten_and_overlaps():
    assert wait(datetime(2026, 11, 9, 9, 30), 45) == 9.5 * H          # ends 10:15: wait to 19:00


def test_weekday_inside_window():
    assert wait(datetime(2026, 11, 11, 14, 0), 20) == 5 * H           # Wednesday 14:00: wait to 19:00


def test_weekday_after_nineteen_ends_in_small_hours():
    assert wait(datetime(2026, 11, 10, 19, 0), 60) == 0
    assert wait(datetime(2026, 11, 10, 23, 30), 60) == 0


def test_saturday_and_sunday_are_free():
    assert wait(datetime(2026, 11, 7, 11, 0), 60) == 0
    assert wait(datetime(2026, 11, 8, 15, 0), 60) == 0


@pytest.mark.parametrize('failed', ['art', 'mm', None])
def test_gate_iv_records_each_group_independently(tmp_path, monkeypatch, failed):
    monkeypatch.setattr(P, 'OUT', tmp_path)

    def check(group):
        assert group != failed, f'{group} failed'
        return {f'{group}_size': 1}

    monkeypatch.setattr(tcl_data, 'gate_iv_art', lambda: check('art'))
    monkeypatch.setattr(tcl_data, 'gate_iv_mm', lambda: check('mm'))
    assert R.cmd_gate_iv() == (1 if failed else 0)
    doc = json.loads(R.gate_path('gate_iv').read_text())
    assert {g: doc[g] for g in R.GROUPS} == {g: g != failed for g in R.GROUPS}
    assert doc['sizes'] == {f'{g}_size': 1 for g in R.GROUPS if g != failed}
