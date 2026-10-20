"""Queue failure propagation, V1 and isolated write destinations (CPU only)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_run
import ct_smoke


@pytest.mark.parametrize('name', ['ct_queue_stage0.sh', 'ct_queue_stage1.sh', 'ct_queue_cpu.sh'])
def test_queue_shell_syntax(name):
    subprocess.run(['bash', '-n', str(HERE / name)], check=True)


def test_stage1_stops_at_missing_gate_and_records_one_failed_step(tmp_path):
    # Run the actual shell queue against a copied source-only queue directory.
    root = tmp_path / 'round'
    root.mkdir()
    for name in ('ct_queue_common.sh', 'ct_queue_stage1.sh', 'ct_stage0.py'):
        (root / name).write_bytes((HERE / name).read_bytes())
    result = subprocess.run(['bash', str(root / 'ct_queue_stage1.sh')], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout.splitlines()[0].startswith('require_stage0=1 seconds=')
    status = (root / 'outputs/queue_stage1_status.txt').read_text()
    assert 'require_stage0=1' in status and 'DONE failed=1' in status
    assert not (root / 'outputs/held').exists()
    assert list((root / 'outputs/logs').glob('*.log')) == [root / 'outputs/logs/stage1_require_stage0.log']


def test_smoke_refuses_production_shared_existing_and_parent_roots(tmp_path):
    for root in (HERE / 'outputs', HERE, Path('/data/new-smoke'), Path('/project/CoSiR/src/test'), tmp_path):
        with pytest.raises(AssertionError):
            ct_smoke.scratch_root(root)
    assert ct_smoke.scratch_root(tmp_path / 'new-smoke') == tmp_path / 'new-smoke'


def test_full_adapter_refuses_cpu_before_data_or_training():
    with pytest.raises(AssertionError, match='GPU'):
        ct_run.main(['d2', '--stage0-training', '--device', 'cpu'])


def test_images_adapter_routes_both_checks_and_writer_away_from_shared_cache(tmp_path, monkeypatch):
    import ct_extract
    import ct_stage0
    monkeypatch.setattr(ct_stage0, 'require_pass', lambda *args: None)
    monkeypatch.setattr(ct_run, 'configure', lambda path: tmp_path)
    writer = ct_extract.write_store
    captured = {}

    def extract(argv):
        captured['check_root'] = ct_extract.OUTPUT
        captured['write_root'] = ct_extract.write_store.keywords['root']
        captured['held_list'] = ct_extract.HELD_LIST
    monkeypatch.setattr(ct_extract, 'main', extract)
    # Restore all adapter globals in the shared pytest process afterward.
    monkeypatch.setattr(ct_extract, 'OUTPUT', ct_extract.OUTPUT)
    monkeypatch.setattr(ct_extract, 'HELD_LIST', ct_extract.HELD_LIST)
    monkeypatch.setattr(ct_extract, 'write_store', writer)
    ct_run.main(['images'])
    assert captured == dict(check_root=tmp_path / 'targets', write_root=tmp_path / 'targets',
                            held_list=tmp_path / 'held/held_paintings.json')


def test_nonfinite_paper_held_output_is_preserved_for_v4_collapse(tmp_path, monkeypatch):
    import numpy as np
    import ct_held
    import ct_percept as pt
    monkeypatch.setattr(pt, 'OUTPUTS', tmp_path)
    names = [f'painting-{i:05d}' for i in range(12281)]
    sha = ct_held.names_sha256(names)
    held_list = tmp_path / 'held/held_paintings.json'
    held_list.parent.mkdir(parents=True)
    held_list.write_text(json.dumps(dict(names=names, sha256=sha)))
    config = dict(settings='paper', cut='k40', arm='P1', seed=0, smoke=False,
                  device='cuda', mapper='attention', input_dim=512, eval_paintings=5000)
    mapper = pt.build_mapper(config, 40)
    path = pt.stage2_path('paper', 'k40', 'P1', 0)
    pt.save_output(path, dict(eval_prob=np.full((5000, 40), .5, dtype=np.float32)),
                   dict(mapper=pt.cpu_state(mapper)), dict(config=config, K=40))
    source = pt.verify_output(path)
    held_path = tmp_path / 'held/percept/paper__k40/P1__s0.npz'
    held_path.parent.mkdir(parents=True)
    prob = np.full((12281, 40), .5, dtype=np.float32)
    prob[0, 0] = np.nan
    np.savez_compressed(held_path, prob=prob)
    meta = dict(config=config, weights_sha256=source['weights_sha256'], held_sha256=sha,
                rows=len(names), inference_device='cuda', array_sha256s=dict(prob=pt.pc_stage1.array_sha256(prob)))
    held_path.with_suffix('.json').write_text(json.dumps(meta))
    before = held_path.read_bytes()
    assert ct_run.frozen_held(tmp_path) == {('paper', 'k40', 'P1', 0)}
    assert held_path.read_bytes() == before
