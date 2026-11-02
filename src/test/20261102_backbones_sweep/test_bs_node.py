"""bs_node.part_a failure logic (final review F2) and the estimate's smoke note (CPU, units stubbed; no training)."""
import json

import pytest

import bs_node as N
import bs_paths as P


@pytest.fixture
def node(tmp_path, monkeypatch):
    monkeypatch.setattr(N, 'NODE', tmp_path / 'node')
    (tmp_path / 'node').mkdir()
    monkeypatch.setattr(N, 'PhotoData', lambda: None)
    return tmp_path / 'node'


def stub_units(monkeypatch, fail_c=(), fail_p=()):
    calls = []

    def unit(kind, fails):
        def fn(enc, seed, device, *pdata):
            calls.append((kind, seed))
            lab = f'{kind}__{enc}__s{seed}'
            return dict(label=lab, status='failed' if seed in fails else 'ok', errors=['boom'] if seed in fails else [])
        return fn
    monkeypatch.setattr(N, 'unit_cstar_part_a', unit('cstar', fail_c))
    monkeypatch.setattr(N, 'unit_xphoto_part_a', unit('xphoto', fail_p))
    return calls


def doc_of(node, enc='siglip2'):
    return json.loads((node / f'part_a_{enc}.json').read_text())


def test_photo_failure_stops_only_photo_units(node, monkeypatch):
    calls = stub_units(monkeypatch, fail_p=(1,))
    assert N.part_a('siglip2', 'cpu') == 1
    d = doc_of(node)
    assert d['status'] == 'failed' and d['photo_failed']['label'] == 'xphoto__siglip2__s1' and d['failed'] is None
    assert [c for c in calls if c[0] == 'cstar'] == [('cstar', s) for s in range(5)]          # every X-c* seed still trains
    assert [c for c in calls if c[0] == 'xphoto'] == [('xphoto', 0), ('xphoto', 1)]            # later X photo units skipped
    assert sum(x.startswith('cstar__') for x in d['done']) == 5 and sum(x.startswith('xphoto__') for x in d['done']) == 1


def test_cstar_failure_stops_the_part(node, monkeypatch):
    calls = stub_units(monkeypatch, fail_c=(2,))
    assert N.part_a('siglip2', 'cpu') == 1
    d = doc_of(node)
    assert d['failed']['label'] == 'cstar__siglip2__s2' and d['photo_failed'] is None
    assert calls[-1] == ('cstar', 2) and ('xphoto', 2) not in calls and ('cstar', 3) not in calls


def test_all_ok(node, monkeypatch):
    stub_units(monkeypatch)
    assert N.part_a('siglip2', 'cpu') == 0
    assert doc_of(node)['status'] == 'ok' and len(doc_of(node)['done']) == 10


def test_estimate_says_when_it_uses_smoke_units(tmp_path, capsys):
    units = tmp_path / 'units.jsonl'
    units.write_text(json.dumps(dict(kind='cstar', status='ok', seconds=10.0, smoke=True)) + '\n')
    out = N.estimate(units_path=units, print_out=True)
    assert 'from smoke units (2 epochs, unscaled): full runs take longer' in capsys.readouterr().out
    assert any('smoke units' in n for n in out['notes'])
    units.write_text(json.dumps(dict(kind='cstar', status='ok', seconds=10.0, smoke=False)) + '\n')
    assert not any('smoke units' in n for n in N.estimate(units_path=units, print_out=False)['notes'])
