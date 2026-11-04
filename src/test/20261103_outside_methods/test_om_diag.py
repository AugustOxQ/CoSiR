"""Synthetic diagnostics: preserve three start values and disclose independent c* seed draws."""
import numpy as np
import pytest

import om_diag as diag
import om_paths as op


class Fits:
    y = np.repeat(np.arange(9), 2)
    parts = np.stack([y, np.zeros_like(y), np.roll(y, 1)])

    def labels(self, *args):
        return self.y

    def tpips_parts(self, *args):
        return self.parts

    clay_parts = tpips_parts
    turtle_plain = tpips_parts

    def turtle_cstar(self, *args):
        return np.repeat(self.y[None, None], 5, axis=0)

    def cstar_parts(self, *args):
        return np.repeat(self.parts[None], 5, axis=0)


def test_sweep_retains_start_spread(tmp_path, monkeypatch):
    monkeypatch.setattr(op, 'OUT', tmp_path)
    monkeypatch.setattr(op, 'R17_OUT', tmp_path / 'r17')
    doc = diag.sweep('mmart', ['emotion'], ['qwen'], [9], ld=Fits())
    assert not doc['missing']
    for row in doc['rows']:
        if row['seeded']:
            assert len(row['per_run']) == 5
        else:
            assert len(row['per_run']) == 3
            assert row['per_run'][0] == pytest.approx(1)
            assert row['per_run'][1] == pytest.approx(0)
            assert row['min'] <= 0
            assert row['max'] == pytest.approx(1)
            assert row['mean'] == pytest.approx(np.mean(row['per_run']))


def test_adds_discloses_independent_seed_draws(tmp_path, monkeypatch):
    monkeypatch.setattr(op, 'OUT', tmp_path)
    monkeypatch.setattr(diag.ns_clu, 'WORKERS', 1)
    monkeypatch.setattr(diag.ns_clu_cells, '_comp_parts', lambda *args: Fits.parts)
    doc = diag.adds('mmart', ['emotion'], ['qwen'], [9], ld=Fits(), draws=3)
    assert not doc['missing']
    assert 'seeds are drawn twice' in doc['note']
    assert 'rows only' in doc['note'] and 'size not measured' in doc['note']
