"""Tests of the queue plumbing fixed after the final review (parts A and H): the fallback moves the node's stage files aside,
stage 4 under F = G runs its two CPU steps, and the ablation queues gate the re-derivation behind F also when F = G.

The queues run on a scratch outputs folder with stand-ins: RS_PYTHON is a wrapper that answers `rs_queue_names.py final|arms`,
echoes for every other round script (no training, no scoring) and runs the real Python for `-I -c` (the `jget` reads); `date` is
a stub on PATH for the fallback's folder stamp. No GPU, no cluster command."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STAMP = '20261028-0100'

WRAPPER = """#!/usr/bin/env bash
case "$*" in
    -I\\ *) exec {python} "$@" ;;
    *rs_queue_names.py*final*) echo G; exit 0 ;;
    *rs_queue_names.py*arms*) echo "direct G-no_direct"; exit 0 ;;
    *rs_sweep.py*score*) echo "stub $*"; exit "${{STUB_SCORE_EXIT:-0}}" ;;
    *rs_*.py*) echo "stub $*"; exit 0 ;;
esac
exec {python} "$@"
"""

DATE = """#!/usr/bin/env bash
if [[ "$1" == '+%Y%m%d-%H%M' ]]; then echo {stamp}; else exec /bin/date "$@"; fi
"""


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


@pytest.fixture
def world(tmp_path):
    out = tmp_path / 'outputs'
    for gate in ('v1a_train', 'v1a_scorer', 'v3a', 'v4a_train', 'v4a_scorer'):
        put(out / 'gates' / f'{gate}.json', {'pass': True})
    bindir = tmp_path / 'bin'
    put(bindir / 'date', DATE.format(stamp=STAMP))
    put(tmp_path / 'python', WRAPPER.format(python=sys.executable))
    for p in (bindir / 'date', tmp_path / 'python'):
        p.chmod(0o755)
    return dict(out=out, tmp=tmp_path, bin=bindir)


def run(world, queue, dry=False, **env_extra):
    env = dict(os.environ, RS_OUTPUTS=str(world['out']), RS_PYTHON=str(world['tmp'] / 'python'), RS_DEVICE='cpu',
               RS_GPU_LOCK=str(world['tmp'] / 'gpu.lock'), RS_FORCE_HOUR='1', RS_PENDING=str(world['tmp'] / 'pending.md'),
               PATH=f'{world["bin"]}:{os.environ["PATH"]}')
    env.pop('RS_DRY', None)
    if dry:
        env['RS_DRY'] = '1'
    env.update(env_extra)
    proc = subprocess.run(['bash', str(HERE / f'rs_queue_{queue}.sh')], env=env, capture_output=True, text=True, timeout=200)
    return proc.returncode, proc.stdout + proc.stderr


def node_files(world, text='{"outcome": "GRID-4"}'):
    put(world['out'] / 'sweep' / 'node402' / 'stage1.json', text)
    put(world['out'] / 'scores' / 'node402' / 'rederive_partitions.json', {'pass': True})


# ---- A1: the fallback moves the node's stage files aside ---------------------------------------------------------------

def test_fallback_moves_node_stage_files_aside(world):
    node_files(world)
    rc, text = run(world, 'fallback', STUB_SCORE_EXIT='1')   # stops at G0's score: only the first step matters here
    out = world['out']
    moved = out / 'aside' / f'node402_{STAMP}' / 'sweep' / 'node402' / 'stage1.json'
    assert moved.read_text() == '{"outcome": "GRID-4"}'
    assert not (out / 'sweep' / 'node402').exists()
    assert (out / 'scores' / 'node402' / 'rederive_partitions.json').exists()    # no reader counts scores/node402
    assert 'rs_node_log.md entry' in text and f'aside/node402_{STAMP}/sweep/node402' in text
    assert 'RS_FALLBACK_STOP' in text and 'G0' in text      # it went on to the next step, which the stand-in failed


def test_fallback_second_run_refuses_on_an_existing_target(world):
    node_files(world)
    run(world, 'fallback', STUB_SCORE_EXIT='1')
    node_files(world, '{"outcome": "SWEEP-NONE"}')      # the node wrote something again: the same minute, the same target
    rc, text = run(world, 'fallback', STUB_SCORE_EXIT='1')
    out = world['out']
    assert rc == 1 and 'RS_FALLBACK_STOP the target' in text and 'exists' in text
    assert (out / 'sweep' / 'node402' / 'stage1.json').read_text() == '{"outcome": "SWEEP-NONE"}'      # nothing moved
    assert (out / 'aside' / f'node402_{STAMP}' / 'sweep' / 'node402' / 'stage1.json').read_text() == '{"outcome": "GRID-4"}'
    assert 'score_g0' not in text      # it stopped before any step


def test_fallback_without_node_files_moves_nothing(world):
    rc, text = run(world, 'fallback', STUB_SCORE_EXIT='1')
    assert not (world['out'] / 'aside').exists() and 'rs_node_log.md entry' not in text


def test_fallback_dry_run_moves_nothing_and_says_so(world):
    node_files(world)
    rc, text = run(world, 'fallback', dry=True)
    assert (world['out'] / 'sweep' / 'node402' / 'stage1.json').exists() and not (world['out'] / 'aside').exists()
    assert 'dry: would move' in text and f'aside/node402_{STAMP}/sweep/node402' in text


def test_the_readers_that_decide_f_ignore_the_moved_node_files(world):
    import rs_full
    out = world['out']
    node_files(world)
    put(out / 'sweep' / 'local' / 'stage1.json', {'outcome': 'SWEEP-NONE'})
    assert rs_full.sweep_ended_without_candidates(out) is None      # a node GRID-n beside a local SWEEP-NONE: F unresolved
    run(world, 'fallback', STUB_SCORE_EXIT='1')
    assert rs_full.sweep_ended_without_candidates(out) == {'local': 'SWEEP-NONE'}


# ---- H2 and A3: stage 4 under F = G ------------------------------------------------------------------------------------

def test_stage4_keep_g_runs_the_two_cpu_steps_and_no_training(world):
    put(world['out'] / 'stage3' / 'verdict.json', {'outcome': 'KEEP-G', 'pick': None})
    rc, text = run(world, 'stage4', dry=True)
    assert rc == 0, text
    assert 'DRY verdict: python -u' in text and 'verdict --candidate G' in text
    assert 'DRY v4b: python -u' in text and 'stage4 --outputs' in text and '--candidate G' in text.split('DRY v4b')[1]
    assert text.index('DRY verdict') < text.index('DRY v4b')
    assert 'train_c' not in text and 'held_c' not in text and 'report' not in text and 'analysis_c' not in text
    assert 'gate V4a-scorer' in text and 'gate V3c' in text and 'gate V4a-train' not in text


def test_stage4_sweep_none_gates_v1c_of_the_sweep_machine(world):
    put(world['out'] / 'sweep' / 'local' / 'stage1.json', {'outcome': 'SWEEP-NONE'})
    rc, text = run(world, 'stage4', dry=True)
    assert rc == 0, text
    assert 'sweep/local/rederive_stage1.json' in text and 'rederive_stage2.json' not in text
    assert 'verdict --candidate G' in text and '--candidate G' in text.split('DRY v4b')[1] and 'train_c' not in text


def test_stage4_top_none_gates_both_stages(world):
    put(world['out'] / 'sweep' / 'local' / 'stage1.json', {'outcome': 'GRID-1'})
    put(world['out'] / 'sweep' / 'local' / 'stage2.json', {'outcome': 'TOP-NONE'})
    rc, text = run(world, 'stage4', dry=True)
    assert 'sweep/local/rederive_stage1.json' in text and 'sweep/local/rederive_stage2.json' in text and 'train_c' not in text


def test_stage4_real_run_under_keep_g_stops_without_v3c_and_never_trains(world):
    put(world['out'] / 'stage3' / 'verdict.json', {'outcome': 'KEEP-G', 'pick': None})
    rc, text = run(world, 'stage4')
    assert rc == 1 and 'gate V3c does not hold' in text and 'stub' not in text
    put(world['out'] / 'stage3' / 'rederive_stage3.json', {'pass': True})
    rc, text = run(world, 'stage4')
    assert rc == 0, text
    logs = world['out'] / 'logs'
    assert 'verdict --candidate G' in (logs / 'stage4_verdict.log').read_text()
    v4b = (logs / 'stage4_v4b.log').read_text()
    assert 'rs_vh.py stage4' in v4b and '--candidate G' in v4b
    assert sorted(p.name for p in logs.iterdir()) == ['stage4_v4b.log', 'stage4_verdict.log']      # no training, held, report or analysis step
    assert 'verdict=0' in text and 'v4b=0' in text and 'RS_STAGE4_DONE failed=0' in text


def test_stage4_v4b_failure_states_no_outcome(world):
    put(world['out'] / 'stage3' / 'verdict.json', {'outcome': 'KEEP-G', 'pick': None})
    put(world['out'] / 'stage3' / 'rederive_stage3.json', {'pass': True})
    wrapper = world['tmp'] / 'python'
    wrapper.write_text(wrapper.read_text().replace('*rs_*.py*)', '*rs_vh.py*) echo "stub $*"; exit 1 ;;\n    *rs_*.py*)'))
    rc, text = run(world, 'stage4')
    assert rc == 1 and 'V4b (rs_vh stage4) disagrees with the verdict on G: no stage 4 outcome is stated' in text


def test_stage4_pick_still_trains_and_the_v4a_replay_path_is_not_under_full(world):
    put(world['out'] / 'stage3' / 'verdict.json', {'outcome': 'PICK', 'pick': 'k6=0.0005'})
    rc, text = run(world, 'stage4', dry=True)
    assert rc == 0 and 'DRY train_c' in text and 'train --config k6=0.0005' in text and 'DRY analysis_c' in text
    assert 'dry: pending row for ' + str(world['out'] / 'regression' / 'v4a_run') in text and '/full/regression' not in text


# ---- A2: the ablation queues gate the re-derivation behind F = G --------------------------------------------------------

@pytest.mark.parametrize('queue', ['ablation_dev', 'ablation_held'])
def test_ablation_queue_gates_v3c_under_keep_g(world, queue):
    put(world['out'] / 'stage3' / 'verdict.json', {'outcome': 'KEEP-G', 'pick': None})
    rc, text = run(world, queue)
    assert rc == 1 and 'gate V3c does not hold' in text and 'rs_train.py' not in text and 'rs_full.py' not in text
    put(world['out'] / 'stage3' / 'rederive_stage3.json', {'pass': True})
    rc, text = run(world, queue)
    assert 'gate V3c holds' in text and 'RS_%s_STOP gate' % queue.upper() not in text


@pytest.mark.parametrize('queue', ['ablation_dev', 'ablation_held'])
def test_ablation_queue_gates_v1c_under_sweep_none(world, queue):
    put(world['out'] / 'sweep' / 'local' / 'stage1.json', {'outcome': 'SWEEP-NONE'})
    rc, text = run(world, queue)
    assert rc == 1 and 'gate V1c-stage1 does not hold' in text and 'rederive_stage1.json' in text
    assert 'rs_train.py' not in text and 'rs_full.py' not in text
    put(world['out'] / 'sweep' / 'local' / 'rederive_stage1.json', {'pass': False})
    rc, text = run(world, queue)
    assert rc == 1 and 'gate V1c-stage1 does not hold' in text         # a failing re-derivation stops as well
    put(world['out'] / 'sweep' / 'local' / 'rederive_stage1.json', {'pass': True})
    rc, text = run(world, queue)
    assert 'gate V1c-stage1 holds' in text and 'V1c-stage2' not in text      # stage 2 did not run


@pytest.mark.parametrize('queue', ['ablation_dev', 'ablation_held'])
def test_ablation_queue_gates_both_v1c_parts_under_top_none(world, queue):
    put(world['out'] / 'sweep' / 'local' / 'stage1.json', {'outcome': 'GRID-1'})
    put(world['out'] / 'sweep' / 'local' / 'stage2.json', {'outcome': 'TOP-NONE'})
    put(world['out'] / 'sweep' / 'local' / 'rederive_stage1.json', {'pass': True})
    rc, text = run(world, queue)
    assert rc == 1 and 'gate V1c-stage1 holds' in text and 'gate V1c-stage2 does not hold' in text
    put(world['out'] / 'sweep' / 'local' / 'rederive_stage2.json', {'pass': True})
    rc, text = run(world, queue)
    assert 'gate V1c-stage2 holds' in text


@pytest.mark.parametrize('queue', ['ablation_dev', 'ablation_held'])
def test_ablation_dry_run_notes_all_the_alternatives(world, queue):
    rc, text = run(world, queue, dry=True)
    assert rc == 0 and 'gate V3c needs pass true' in text and 'rederive_stage1.json' in text and 'rederive_stage2.json' in text
