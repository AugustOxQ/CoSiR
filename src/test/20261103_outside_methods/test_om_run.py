"""Runner guards: current committed status, isolated smoke environments and complete range records. Synthetic only."""
import copy
from argparse import Namespace

import pytest

import om_cells
import om_paths as op
import om_run as run


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
