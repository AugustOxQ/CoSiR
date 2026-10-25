"""Gates of the stage driver on synthetic output folders (no data, no GPU)."""
import json
import os
import time

import pytest

import rb_run


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def reg_items(ok=True):
    items = {k: {'pass': True} for names in rb_run.EXPECTED.values() for k in names}
    items.update({'scorer/A_e/x': {'pass': True}, 'scorer/partitions/B1': {'pass': True}})
    return items


def test_stage1_needs_passed_stage0(tmp_path):
    put(tmp_path / 'stage0.json', {'pass': False, 'smoke': False})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage1(tmp_path, False)
    put(tmp_path / 'stage0.json', {'pass': True, 'smoke': False})
    rb_run.gate_stage1(tmp_path, False)


def test_vp1_needs_agree_after_pick(tmp_path):
    put(tmp_path / 'part1/pick.json', {})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_vp1(tmp_path, False)                      # no rederive file
    put(tmp_path / 'part1/rederive_vp1.json', {'agree': False})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_vp1(tmp_path, False)
    put(tmp_path / 'part1/rederive_vp1.json', {'agree': True})
    rb_run.gate_vp1(tmp_path, False)
    old = time.time() - 100
    os.utime(tmp_path / 'part1/rederive_vp1.json', (old, old))
    with pytest.raises(rb_run.Stop):
        rb_run.gate_vp1(tmp_path, False)                      # stale


@pytest.mark.parametrize('drop', ['student/genre', 'matched/qwen_s0/config', 'scorer/d', 'scorer/A_e/x', 'scorer/partitions/B1'])
def test_regression_needs_every_item(tmp_path, drop):
    items = reg_items()
    put(tmp_path / 'part2/regression/stage2.json', {'pass': True, 'items': items})
    assert 'items' in rb_run.regression_complete(tmp_path, 'stage2') or True
    del items[drop]
    put(tmp_path / 'part2/regression/stage2.json', {'pass': True, 'items': items})
    with pytest.raises(rb_run.Stop):
        rb_run.regression_complete(tmp_path, 'stage2')


def test_regression_failing_item(tmp_path):
    items = reg_items()
    items['scorer/d'] = {'pass': False}
    put(tmp_path / 'part2/regression/stage3a.json', {'pass': False, 'items': items})
    with pytest.raises(rb_run.Stop):
        rb_run.regression_complete(tmp_path, 'stage3a')


def test_candidates_and_stage5_scales(tmp_path, monkeypatch):
    put(tmp_path / 'part1/pick.json', {'triggers': {'K4': True, 'K5': False}})
    assert rb_run.candidates(tmp_path, 3) == ['K2', 'K3', 'K4']
    put(tmp_path / 'part2/verdict_stage3.json', {'blocks': {
        'K2': {'outcome_id': 'S1-NO'}, 'K3': {'outcome_id': 'S1-MM'}, 'K4': {'outcome_id': None}}})
    assert rb_run.candidates(tmp_path, 4) == ['K3']
    monkeypatch.setenv('RB_DEFER', 'K3')
    assert rb_run.candidates(tmp_path, 3) == ['K2', 'K4']
    monkeypatch.delenv('RB_DEFER')
    put(tmp_path / 'part2/verdict_stage4.json', {'blocks': {'K3': {'outcome_id': 'S2-IN'}, 'K4': {'outcome_id': None}}})
    models, scales = rb_run.a28_plan(tmp_path)
    assert models == ['K1', 'K2', 'K3', 'K4'] and scales == dict(K1='full', K2='sample', K3='full', K4='sample')


def test_vp3(tmp_path):
    rb_run.vp3(tmp_path, 'R0-5', False)
    for a in ('emotion', 'style', 'genre'):
        put(tmp_path / f'part1/train/R0-5/{a}__s0__P.npz', {})
    rb_run.vp3(tmp_path, 'R0-3', False)                     # a recipe of the plan that is not finished: allowed
    with pytest.raises(rb_run.Stop):
        rb_run.vp3(tmp_path, 'R0-5', False)                 # finished: retraining refused
    rb_run.vp3(tmp_path, 'R0-5', True)                      # forced rerun, logged
    assert len((tmp_path / 'part1/vp3_log.jsonl').read_text().splitlines()) == 3
    with pytest.raises(rb_run.Stop):
        rb_run.vp3(tmp_path, 'R9-9', False)


def test_vp2_cause(tmp_path):
    log = tmp_path / 'l.log'
    log.write_text('Traceback\nAssertionError: R6-5 emotion img: norm error 0.5\n')
    assert 'norm error' in rb_run.vp2_cause(log)
    log.write_text('Traceback\nRuntimeError: CUDA out of memory\n')
    with pytest.raises(rb_run.Stop):
        rb_run.vp2_cause(log)
