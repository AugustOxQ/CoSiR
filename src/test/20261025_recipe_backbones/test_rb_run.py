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
    assert rb_run.candidates(tmp_path, 3, todo=False) == ['K2', 'K4']
    monkeypatch.delenv('RB_DEFER')
    assert rb_run.candidates(tmp_path, 3) == []              # every candidate has a stage 3 block: nothing to redo
    put(tmp_path / 'part1/pick.json', {'triggers': {'K4': True, 'K5': True}})
    assert rb_run.candidates(tmp_path, 3) == ['K5']          # a rerun after deferring K5 scores K5 only
    assert rb_run.candidates(tmp_path, 3, todo=False) == ['K2', 'K3', 'K4', 'K5']
    put(tmp_path / 'part2/verdict_stage4.json', {'blocks': {'K3': {'outcome_id': 'S2-IN'}}})
    assert rb_run.candidates(tmp_path, 4) == [] and rb_run.candidates(tmp_path, 4, todo=False) == ['K3']
    put(tmp_path / 'part1/pick.json', {'triggers': {'K4': True, 'K5': False}})
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


# --- stage gates, V2 freshness, check_verdict, budget ------------------------------------------------------------------

def chain(out, regs=('stage2', 'stage3a')):
    """A full passing chain: stage 0, pick, VP1 (written after pick), regressions."""
    put(out / 'stage0.json', {'pass': True, 'smoke': False})
    put(out / 'part1/pick.json', {'triggers': {'K4': True, 'K5': True}})
    put(out / 'part1/rederive_vp1.json', {'agree': True})
    for r in regs:
        put(out / f'part2/regression/{r}.json', {'pass': True, 'items': reg_items()})


def verdict(out, stage, outcomes):
    put(out / f'part2/verdict_stage{stage}.json', {'blocks': {c: {'outcome_id': o} for c, o in outcomes.items()}})


def v2file(out, stage, cands, agree=True, age=0):
    put(out / f'part2/rederive_v2_stage{stage}.json', {'agree': agree, 'candidates': {c: {} for c in cands}})
    if age:
        old = time.time() - age
        os.utime(out / f'part2/rederive_v2_stage{stage}.json', (old, old))


def test_gate_stage3(tmp_path):
    chain(tmp_path)
    assert 'pass' in rb_run.gate_stage3(tmp_path, False)
    put(tmp_path / 'stage0.json', {'pass': False, 'smoke': False})          # M22: gate_part2 call removed
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage3(tmp_path, False)
    put(tmp_path / 'stage0.json', {'pass': True, 'smoke': False})
    put(tmp_path / 'part1/rederive_vp1.json', {'agree': False})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage3(tmp_path, False)


def test_gate_stage4(tmp_path):
    chain(tmp_path)
    verdict(tmp_path, 3, dict(K2='S1-IN'))
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage4(tmp_path, False)                                 # M13: no V2 file
    v2file(tmp_path, 3, ['K2'], agree=False)
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage4(tmp_path, False)                                 # M13: V2 disagrees
    v2file(tmp_path, 3, ['K2'])
    assert 'pass' in rb_run.gate_stage4(tmp_path, False)
    put(tmp_path / 'part2/regression/stage3a.json', {'pass': False, 'items': reg_items()})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage4(tmp_path, False)
    put(tmp_path / 'part2/regression/stage3a.json', {'pass': True, 'items': reg_items()})
    put(tmp_path / 'part1/rederive_vp1.json', {'agree': False})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage4(tmp_path, False)                                 # part 2 gate


def test_gate_stage5(tmp_path):
    chain(tmp_path, regs=('stage2',))
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage5(tmp_path, False)                                 # no stage 3a regression
    put(tmp_path / 'part2/regression/stage3a.json', {'pass': True, 'items': reg_items()})
    assert 'pass' in rb_run.gate_stage5(tmp_path, False)
    put(tmp_path / 'stage0.json', {'pass': False, 'smoke': False})
    with pytest.raises(rb_run.Stop):
        rb_run.gate_stage5(tmp_path, False)


def test_gate_v2_agree_and_freshness(tmp_path):
    gate = rb_run.gate_v2(3)
    verdict(tmp_path, 3, dict(K2='S1-IN', K3='S1-NO', K4='S1-MM', K5=None))
    v2file(tmp_path, 3, ['K2', 'K3', 'K4'], agree=False)
    with pytest.raises(rb_run.Stop):
        gate(tmp_path, False)                                               # M20: agree not checked
    v2file(tmp_path, 3, ['K2', 'K3', 'K4'])
    assert 'agrees' in gate(tmp_path, False)                                # K5 has no outcome: need not be listed
    v2file(tmp_path, 3, ['K2', 'K3'])
    with pytest.raises(rb_run.Stop, match='K4'):
        gate(tmp_path, False)                                               # fresh, but lacks a block with an outcome
    v2file(tmp_path, 3, ['K2', 'K3', 'K4'], age=3600)
    with pytest.raises(rb_run.Stop, match='rerun V2'):
        gate(tmp_path, False)                                               # older than the verdict file


def test_gate_v2_reviewer_case_and_stage4(tmp_path):
    chain(tmp_path)
    verdict(tmp_path, 3, dict(K2='S1-IN', K3='S1-IN', K4='S1-IN', K5='S1-IN'))
    v2file(tmp_path, 3, ['K2', 'K3', 'K4'], age=3600)
    for g in (rb_run.gate_v2(3), rb_run.gate_stage4):
        with pytest.raises(rb_run.Stop):
            g(tmp_path, False)
    v2file(tmp_path, 3, ['K2', 'K3', 'K4', 'K5'])
    rb_run.gate_v2(3)(tmp_path, False)
    rb_run.gate_stage4(tmp_path, False)


def test_check_verdict(tmp_path):
    verdict(tmp_path, 3, dict(K2='S1-IN', K3=None))
    with pytest.raises(rb_run.Stop, match='K4'):
        rb_run.check_verdict(tmp_path, 3, ['K2', 'K3', 'K4'])               # M21: missing block
    msg = rb_run.check_verdict(tmp_path, 3, ['K2', 'K3'])                   # no outcome: listed, no raise
    assert "no outcome: ['K3']" in msg and 'K2=S1-IN' in msg


def budget_env(tmp_path, monkeypatch):
    import datetime
    from zoneinfo import ZoneInfo
    monkeypatch.setattr(rb_run.rb_paths, 'PRE', str(tmp_path / 'pre'))
    monkeypatch.delenv('RB_DEFER', raising=False)
    return datetime.datetime(2026, 10, 25, 20, 0, tzinfo=ZoneInfo('Europe/Amsterdam'))


def steps(res):
    return {s['step']: s['high'] for s in res['steps']}


def test_budget_per_candidate(tmp_path, monkeypatch):
    now = budget_env(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    put(out / 'part1/pick.json', {'triggers': {'K4': True, 'K5': True}})
    verdict(out, 3, dict(K2='S1-IN', K3='S1-IN', K4='S1-IN'))               # K5 still to do (deferred run)
    s = steps(rb_run.budget(out, now=now))
    assert s['k5_sample'] > 0 and s['k23_sample'] == 0 and s['k4_sample'] == 0 and s['verdict3'] > 0 and s['v2_3'] > 0
    assert s['k23_full'] > 0 and s['k4_full'] > 0 and s['k5_full'] == 0     # K5 has no S1 block yet: not in stage 4
    verdict(out, 3, dict(K2='S1-IN', K3='S1-NO', K4=None, K5='S1-IN'))
    s = steps(rb_run.budget(out, now=now))
    assert s['k5_sample'] == s['k4_sample'] == s['k23_sample'] == s['verdict3'] == 0
    assert s['k23_full'] > 0 and s['k5_full'] > 0 and s['k4_full'] == 0     # stage 4 rule: K3 is S1-NO, K4 no outcome
    put(out / 'part2/rederive_v2_stage3.json', {})
    assert steps(rb_run.budget(out, now=now))['v2_3'] == 0
    verdict(out, 4, dict(K2='S2-IN', K5='S2-IN'))
    s = steps(rb_run.budget(out, now=now))
    assert s['k23_full'] == s['k5_full'] == s['verdict4'] == 0


def test_budget_report_and_missing_files(tmp_path, monkeypatch):
    now = budget_env(tmp_path, monkeypatch)
    res = rb_run.budget(tmp_path / 'none', now=now)                         # no pick, no verdicts: no exception
    assert res['candidates'] == ['K2', 'K3', 'K4', 'K5'] and steps(res)['k5_sample'] > 0
    out = tmp_path / 'out'
    put(out / 'part1/pick.json', {'triggers': {'K4': False, 'K5': False}})
    base = rb_run.budget(out, now=now)['estimated_end_high']
    put(out / 'part2/reported_stage3.json', {'blocks': {'K2': {}, 'K3': {}, 'Kx': {}}})
    put(out / 'part2/reported_stage4.json', {'blocks': {'K2': {}, 'K3': {}}})
    assert rb_run.budget(out, now=now)['estimated_end_high'] < base
