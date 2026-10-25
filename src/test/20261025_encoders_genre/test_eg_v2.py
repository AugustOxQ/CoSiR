"""Tests of eg_v2 (ticket 06): which candidates V2 re-derives, where it reads the arms from, and the comparison with
this round's verdict file (agree, and a forced mismatch). Small synthetic worlds built with test_rb_v2's helpers."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import eg_paths
import eg_v2 as v2
import rb_v2
import test_rb_v2 as rbt


# --- the candidate filter (rule §2.1 order; a block without an outcome is skipped, not re-derived) ---------------------

def write_verdict(path, blocks, stage=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema': 1, 'stage': stage, 'blocks': blocks}))
    return path


GOOD = dict(outcome_id='S1-GO', d=.02)
BAD = dict(outcome_id=None, v_failure='AssertionError: missing output')


def test_ready_candidates_are_the_blocks_with_an_outcome_in_the_rule_order(tmp_path):
    f = write_verdict(tmp_path / 'v.json', {'K5': GOOD, 'K8': BAD, 'K6': GOOD})
    ready, skipped = v2.candidates_with_outcome(f)
    assert ready == ['K6', 'K5']                                 # K6, K7, K8, K5 is the order, not the file's
    assert list(skipped) == ['K8'] and 'missing output' in skipped['K8']['v_failure']


def test_the_default_leaves_out_blocks_of_other_candidates_and_a_named_candidate_needs_its_block(tmp_path):
    f = write_verdict(tmp_path / 'v.json', {'K2': GOOD, 'K7': GOOD})
    assert v2.candidates_with_outcome(f)[0] == ['K7']
    with pytest.raises(AssertionError):
        v2.candidates_with_outcome(f, ['K6'])                    # no block for K6
    with pytest.raises(AssertionError):
        v2.candidates_with_outcome(f, ['K2'])                    # 12-24's candidate: only for the parity check
    assert v2.candidates_with_outcome(f, ['K2'], allow_old=True)[0] == ['K2']
    assert v2.candidates_with_outcome(f, ['K7'])[0] == ['K7']


# --- where the arms are read from ---------------------------------------------------------------------------------------

def test_stage5_b1_and_b6_at_the_full_scale_come_from_12_24s_outputs(tmp_path):
    roots = v2.Roots.default(tmp_path / 'out', frozen=tmp_path / 'frozen')
    full = {p['name']: p['base'] for p in v2.arm_plans('K6', 'a28', 'full', roots)}
    assert full['B1'] == full['B6'] == tmp_path / 'frozen/part2/full/a28'
    assert full['K6'] == full['M6'] == tmp_path / 'out/part2/full/a28'
    sample = {p['name']: p['base'] for p in v2.arm_plans('K6', 'a28', 'sample', roots)}
    assert set(sample.values()) == {tmp_path / 'out/part2/sample/a28'}
    held = {p['name']: p['base'] for p in v2.arm_plans('K6', 'held', 'full', roots)}
    assert held['K6'] == held['M6'] == tmp_path / 'out/part2/full/held'
    assert held['B1'] == Path(roots.ct['full']) / 'held'


def test_the_plan_override_is_scoped_and_leaves_rb_v2_as_it_was(tmp_path):
    before = rb_v2.arm_plans
    roots = v2.Roots.default(tmp_path / 'out', frozen=tmp_path / 'frozen')
    with v2.scoped_plans():
        assert rb_v2.arm_plans is not before
    assert rb_v2.arm_plans is before
    del roots


# --- the comparison with the verdict file: held scale ---------------------------------------------------------------------

def held_world(tmp, candidate='K6'):
    """A synthetic held world with K6 and M6 in place of the toy world's K2 and M2, and the verdict of K6."""
    w = rbt.World(tmp)
    for s in range(5):
        for old, new, key in (('K2', candidate, 'img'), ('M2', 'M' + candidate[1:], 'prob')):
            folder = w.out / 'part2/sample/held'
            name = f'emotion__s{s}' if old == 'K2' else f'P3__s{s}'
            side = json.loads((folder / old / f'{name}.json').read_text())
            rbt.save(folder / new / f'{name}.npz', key, np.load(folder / old / f'{name}.npz')[key], dict(side, arm=new))
    res = rb_v2.run_candidate(candidate, 3, w.roots, draws=8, workers=2)
    rbt.save_partitions(w.out, 'sample', res)
    return w, res


def verdict_for(w, res, candidate='K6', **change):
    block = rbt.stage_block(res, outcome_id='S1-GO', scale='sample')
    block.update(change)
    write_verdict(w.out / 'part2/verdict_stage3.json', {candidate: block})


def test_v2_agrees_with_a_verdict_it_did_not_write_and_records_it(tmp_path):
    w, res = held_world(tmp_path)
    verdict_for(w, res)
    out = v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)
    assert out['agree'] and out['candidates'] == {'K6': []}
    doc = json.loads((w.out / 'part2/rederive_v2_stage3.json').read_text())
    assert doc['agree'] is True and doc['candidates']['K6']['comparison']['agree'] is True
    assert doc['round'] == 'encoders_genre' and doc['rule'] == str(eg_paths.RULE) and doc['code_commit']
    assert doc['candidates']['K6']['partitions']['unchecked'] == 0 and doc['candidates']['K6']['partitions']['checked'] > 0


def test_a_forced_difference_in_d_or_an_interval_bound_makes_it_disagree(tmp_path):
    w, res = held_world(tmp_path)
    verdict_for(w, res, d=res['d'] + 1e-3)
    out = v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)
    assert not out['agree'] and any('d:' in m for m in out['candidates']['K6'])
    doc = json.loads((w.out / 'part2/rederive_v2_stage3.json').read_text())
    assert doc['agree'] is False and doc['candidates']['K6']['comparison']['agree'] is False
    verdict_for(w, res, interval=[res['interval'][0], res['interval'][1] + 1e-3])
    out = v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)
    assert not out['agree'] and any('interval upper' in m for m in out['candidates']['K6'])


def test_a_changed_eligibility_or_a_changed_saved_partition_makes_it_disagree(tmp_path):
    w, res = held_world(tmp_path)
    eligible = {k: list(v) for k, v in res['eligible_seeds'].items()}
    eligible['B2'] = [0, 1, 2, 3, 4]                              # the R-COL record says [0, 1, 3, 4]
    verdict_for(w, res, eligibility=dict(eligible_seeds=eligible))
    assert not v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)['agree']
    verdict_for(w, res)
    f = w.out / 'part2/partitions/sample/K6/emotion__s1__r20.npz'
    bad = np.load(f)['labels'].copy()
    bad[0, 0] += 1
    np.savez(f, labels=bad)
    out = v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)
    assert not out['agree'] and any('K6 s1 r20' in m for m in out['candidates']['K6'])


def test_a_file_with_no_candidate_re_derived_does_not_agree(tmp_path):
    w, res = held_world(tmp_path)
    write_verdict(w.out / 'part2/verdict_stage3.json', {'K6': dict(BAD)})
    out = v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)
    assert out['agree'] is False and out['candidates'] == {} and list(out['skipped']) == ['K6']
    doc = json.loads((w.out / 'part2/rederive_v2_stage3.json').read_text())
    assert doc['agree'] is False and doc['skipped']['K6']['v_failure']


def test_a_rerun_re_derives_every_candidate_again_and_there_is_no_keep_path(tmp_path, monkeypatch):
    """A kept entry would outlive changed outputs or code (finding 1): every run derives every candidate again."""
    w, res = held_world(tmp_path)
    verdict_for(w, res)
    assert v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)['agree']
    calls = []
    real = rb_v2.run_candidate
    monkeypatch.setattr(rb_v2, 'run_candidate', lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    assert v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)['agree'] and calls == ['K6']
    verdict_for(w, res, d=res['d'] + 1e-3)                        # a replaced block: now disagrees
    assert not v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)['agree'] and calls == ['K6', 'K6']
    with pytest.raises(TypeError):
        v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2, force=True)
    with pytest.raises(SystemExit):
        v2.main(['--stage', '3', '--force'])


def test_changed_outputs_after_an_agreeing_run_are_seen_by_the_next_run(tmp_path):
    w, res = held_world(tmp_path)
    verdict_for(w, res)
    assert v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)['agree']
    f = w.out / 'part2/partitions/sample/K6/emotion__s1__r20.npz'
    bad = np.load(f)['labels'].copy()
    bad[0, 0] += 1
    np.savez(f, labels=bad)
    out = v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)
    assert not out['agree'] and any('K6 s1 r20' in m for m in out['candidates']['K6'])


# --- --candidates only for scratch outputs (finding 3) --------------------------------------------------------------------

def test_a_candidate_list_is_refused_for_outputs_outside_the_scratch_root(tmp_path, monkeypatch):
    """`--candidates` would drop the other candidates' entries from the file: allowed only under /tmp/claude-0/."""
    w, res = held_world(tmp_path)
    verdict_for(w, res)
    monkeypatch.setattr(eg_paths, 'SMOKE_ROOT', tmp_path / 'elsewhere')
    with pytest.raises(AssertionError, match='candidates'):
        v2.run_stage(3, w.out, ['K6'], roots=w.roots, draws=8, workers=2)
    with pytest.raises(AssertionError, match='candidates'):
        v2.main(['--stage', '3', '--candidates', 'K6', '--outputs', str(w.out)])
    assert not (w.out / 'part2/rederive_v2_stage3.json').exists()
    monkeypatch.setattr(eg_paths, 'SMOKE_ROOT', tmp_path)
    assert v2.run_stage(3, w.out, ['K6'], roots=w.roots, draws=8, workers=2)['agree']
    assert v2.run_stage(3, w.out, roots=w.roots, draws=8, workers=2)['agree']       # no list: always allowed


# --- stage 5 -------------------------------------------------------------------------------------------------------------

def test_stage5_reads_b1_and_b6_at_the_full_scale_from_the_frozen_outputs(tmp_path):
    w = rbt.A28World(tmp_path)
    w.write_model('K6', 'full')
    frozen = tmp_path / 'frozen'
    for b in ('B1', 'B6'):
        (frozen / 'part2/full/a28').mkdir(parents=True, exist_ok=True)
        shutil.move(str(w.out / 'part2/full/a28' / b), str(frozen / 'part2/full/a28' / b))
    roots = v2.Roots(outputs=w.roots.outputs, ct=w.roots.ct, held_sha='x', rcol={}, annotations=[],
                     a28=w.roots.a28, a28_paintings=w.n, frozen=frozen)
    with v2.scoped_plans():
        res = rb_v2.run_candidate('K6', 5, roots, draws=6, workers=2, scale='full')
    rbt.save_partitions(w.out, 'a28_full', res)
    block = w.block(res, scale='full')
    w.write_verdicts({'K6': block}, stage4={'K6': dict(outcome_id='S2-GO')})
    out = v2.run_stage(5, w.out, roots=roots, draws=6, workers=2)
    assert out['agree'] and out['candidates'] == {'K6': []}
    doc = json.loads((w.out / 'part2/rederive_v2_stage5.json').read_text())
    assert doc['candidates']['K6']['scale'] == 'full'
    w.write_verdicts({'K6': dict(block, scale='sample')}, stage4={'K6': dict(outcome_id='S2-GO')})
    out = v2.run_stage(5, w.out, roots=roots, draws=6, workers=2)
    assert not out['agree'] and any('scale' in m for m in out['candidates']['K6'])
