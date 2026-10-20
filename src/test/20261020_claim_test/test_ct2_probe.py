"""CPU execution of the GPU probe's real workloads and JSON/OOM contract."""
import json

import pytest
import torch

import ct2_probe as probe


def test_probe_plan_covers_both_dec_versions_and_every_mapper_cut():
    items = probe.probe_items()
    dec = [item for item in items if item['stage'] == 'joint']
    assert {(x['version'], x['chunked']) for x in dec} == {
        (version, chunked) for version in ('clip_roberta', 'qwen_qwen') for chunked in (False, True)}
    mappers = [item for item in items if item['stage'] == 'mapper']
    assert {(x['settings'], x['arm'], x['cut'], x['topics']) for x in mappers} == {
        ('core', 'P2', 'k40', 40), ('core', 'P3', 'k40', 40),
        ('paper', 'P1', 'strict', 100), ('paper', 'P4', 'strict', 100),
        ('paper', 'P1', 'k40', 40), ('paper', 'P4', 'k40', 40)}
    assert probe.FIT_PAINTINGS == 42969 and probe.FIT_PAIRS == 216107
    assert probe.DEV_PAINTINGS == 6152


@pytest.mark.parametrize('item', range(10))
def test_probe_executes_real_epoch_with_small_cpu_inputs(item, capsys):
    result = probe.run_item(probe.probe_items()[item], 'cpu', fit_pairs=37, fit_paintings=7, dev_paintings=3)
    assert result['status'] == 'ok'
    assert result['epochs'] == 1
    assert result['fit_pairs'] == 37 and result['fit_paintings'] == 7 and result['dev_paintings'] == 3
    assert result['peak_allocated_bytes'] is None
    assert result['seconds'] > 0
    assert capsys.readouterr().out == '', 'stdout is reserved for JSON lines'
    json.dumps(result, allow_nan=False)


def test_oom_is_a_json_result_and_other_errors_remain_errors():
    def oom():
        raise torch.cuda.OutOfMemoryError('synthetic OOM')
    result = probe.measure({}, 'cpu', oom)
    assert result['status'] == 'out_of_memory'
    assert 'synthetic OOM' in result['error']
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='bad input'):
        probe.measure({}, 'cpu', lambda: (_ for _ in ()).throw(ValueError('bad input')))


def test_cli_prints_one_json_line_per_item(capsys):
    results = probe.main(['--device', 'cpu', '--smoke'])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(results) == 10
    assert [json.loads(line) for line in lines] == results
    assert all(result['status'] == 'ok' and result['smoke'] for result in results)


def test_cuda_cli_spelling_resolves_device_index_before_driver_calls(monkeypatch):
    # Fake only the CUDA driver boundary: this test must not initialize a GPU.
    def selected(device):
        assert torch.device(device) == torch.device('cuda:0')
    monkeypatch.setattr(torch.cuda, 'set_device', selected)
    monkeypatch.setattr(torch.cuda, 'synchronize', selected)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', selected)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 12345)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 23456)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'test GPU')
    result = probe.measure({}, 'cuda', lambda: None)
    assert result['device'] == 'cuda:0'
    assert result['peak_allocated_bytes'] == 12345 and result['peak_reserved_bytes'] == 23456
    assert result['gpu_name'] == 'test GPU'
