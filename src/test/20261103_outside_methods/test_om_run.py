"""Runner guards, isolated stage 2 smoke and complete range records. Synthetic only, no real records or stores."""
import copy
import json
from argparse import Namespace

import pytest

import om_cells
import om_paths as op
import om_run as run
import om_stage1 as s1


@pytest.mark.parametrize('command', ['assemble', 'labels'])
@pytest.mark.parametrize('only', [None, 'SC5/mmart/qwen'])
@pytest.mark.parametrize('key', ['gates.iv', 'stage1.tpips/mmart', 'r17.sha256'])
def test_stale_status_refuses_before_dispatch(monkeypatch, command, only, key):
    committed = dict(gates={'iv': 'ok'}, stage1={'tpips/mmart': 'ok'}, r17=dict(path='old', sha256='old'))
    current = copy.deepcopy(committed)
    section, field = key.split('.', 1)
    current[section][field] = 'changed'
    monkeypatch.setattr(op, 'SMOKE', 1)
    monkeypatch.setattr(op, 'read_committed_record', lambda name: committed)
    monkeypatch.setattr(run, 'build_status', lambda write: current)
    with pytest.raises(SystemExit, match=key + '.*run status and commit'):
        getattr(run, 'cmd_' + command)(Namespace(only=only))


@pytest.mark.parametrize('command', ['assemble', 'labels'])
def test_current_status_ignores_checkout_path(monkeypatch, command):
    committed = dict(gates={'iv': 'ok'}, stage1={}, r17=dict(path='old', sha256='same'))
    current = copy.deepcopy(committed)
    current['r17']['path'] = 'new'
    monkeypatch.setattr(op, 'read_committed_record', lambda name: committed)
    monkeypatch.setattr(run, 'build_status', lambda write: current)
    monkeypatch.setattr(om_cells, 'cmd_' + command, lambda args: 42)
    assert getattr(run, 'cmd_' + command)(Namespace(only=None)) == 42


@pytest.mark.parametrize('entry', [run.main, om_cells.main])
@pytest.mark.parametrize('smoke,outputs', [(None, 'scratch'), ('0', 'scratch'), ('1', None), ('1', 'default'),
                                         ('1', 'alias')])
def test_cli_refuses_mixed_environment(monkeypatch, tmp_path, entry, smoke, outputs):
    monkeypatch.delenv('OM_SMOKE', raising=False)
    monkeypatch.delenv('OM_OUTPUTS', raising=False)
    if smoke is not None:
        monkeypatch.setenv('OM_SMOKE', smoke)
    if outputs is not None:
        if outputs == 'alias':
            alias = tmp_path / 'alias'
            alias.symlink_to(op.DEFAULT_OUT, target_is_directory=True)
            outputs = str(alias)
        elif outputs == 'default':
            outputs = str(op.DEFAULT_OUT)
        monkeypatch.setenv('OM_OUTPUTS', outputs)
    with pytest.raises(SystemExit, match='OM_'):
        entry(['list'])


@pytest.mark.parametrize('entry', [run.main, om_cells.main])
@pytest.mark.parametrize('smoke', [False, True])
def test_cli_accepts_real_or_isolated_smoke(monkeypatch, tmp_path, entry, smoke):
    monkeypatch.delenv('OM_SMOKE', raising=False)
    monkeypatch.delenv('OM_OUTPUTS', raising=False)
    if smoke:
        monkeypatch.setenv('OM_SMOKE', '1')
        monkeypatch.setenv('OM_OUTPUTS', str(tmp_path))
    assert entry(['list']) == 0


@pytest.mark.parametrize('count', [1, 139, 140, 141])
def test_stage1_records_requires_complete_range_count(monkeypatch, tmp_path, count):
    monkeypatch.setattr(op, 'OUT', tmp_path)
    monkeypatch.setattr(op, 'record_path', lambda name: tmp_path / 'committed' / name)
    (tmp_path / 'stage1').mkdir()
    doc = {str(i): {'r': 0.01} for i in range(count)}
    (tmp_path / 'stage1' / 'om_ranges.json').write_text(json.dumps(doc))
    if count == len(s1.range_keys()):
        run.stage1_records()
        assert json.loads((tmp_path / 'committed' / 'stage1_om_ranges.json').read_text()) == doc
    else:
        with pytest.raises(AssertionError, match='range.*keys'):
            run.stage1_records()
        assert not (tmp_path / 'committed').exists()


def test_stage2_cli_scores_and_resumes_in_isolated_smoke(monkeypatch, tmp_path):
    from test_om_cells import FakeLoaders, good_picks
    monkeypatch.setenv('OM_SMOKE', '1')
    monkeypatch.setenv('OM_OUTPUTS', str(tmp_path))
    monkeypatch.setattr(op, 'SMOKE', 1)
    monkeypatch.setattr(op, 'OUT', tmp_path)
    monkeypatch.setattr(op, 'read_committed_record', lambda name: good_picks() if 'picks' in name else {})
    monkeypatch.setattr(om_cells, 'Loaders', FakeLoaders)
    monkeypatch.setattr(om_cells.ns_clu, 'WORKERS', 1)
    args = ['stage2', '--sc', 'SC5', '--set', 'mmart', '--enc', 'qwen', '--draws', '3']
    assert run.main(args) == 0
    part = tmp_path / 'stage2/parts/SC5__mmart__qwen.json'
    cells = json.loads(part.read_text())
    assert len(cells) == 14 and all(c['n_draws'] == 3 and c['missing'] is None for c in cells)
    before = part.stat().st_mtime_ns
    assert run.main(args) == 0
    assert part.stat().st_mtime_ns == before
    assert json.loads((tmp_path / 'steps/stage2_SC5_mmart_qwen.json').read_text())['status'] == 'ok'
