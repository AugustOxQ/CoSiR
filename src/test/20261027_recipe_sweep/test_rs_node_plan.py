"""Tests of rs_node_plan.sh (the node route of stages 1 and 2): the deadlines as date-times, the V1a gate of stage 1, and the
move of the stage 1 rederive_partitions.json before the second fetch of stage 2.

The plan runs in real mode here, on a scratch outputs folder, with stand-ins for everything that would touch the cluster or
the scorers: RS_CLUSTER (a script that prints and succeeds), RS_SYNC_PY (a copy of das6_sync_rs.fetch's rule: a differing local
file refuses the fetch, else the "node" file is copied), RS_PYTHON (a wrapper that echoes for rs_sweep.py and rs_vp.py and runs
the real Python for everything else). No cluster command runs."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'rs_node_plan.sh'

# The cluster stand-in: prints its arguments; the `vp` job "rewrites the node's file" (NODE_AFTER over NODE_FILE), as
# `rs_vp partitions` does on the node.
CLUSTER = """#!/usr/bin/env bash
echo "cluster $*"
[[ "$1" == launch && "$*" == *-vp* && -n "${NODE_AFTER:-}" ]] && cp "$NODE_AFTER" "$NODE_FILE"
exit 0
"""

WRAPPER = """#!/usr/bin/env bash
case "$*" in
    *rs_sweep.py*|*rs_vp.py*) echo "stub $*"; exit 0 ;;
esac
exec {python} "$@"
"""

# The stand-in for das6_sync_rs.py: `--fetch scores/node402` copies $NODE_DIR/rederive_partitions.json to the scratch outputs
# unless a differing file is there (the real rule: "REFUSING: differing local files").
SYNC = """import os, sys
from pathlib import Path
rel = sys.argv[sys.argv.index('--fetch') + 1]
out = Path(os.environ['RS_OUTPUTS'])
if os.environ.get('SYNC_FAIL_ON'):      # fail the Nth call of this attempt (the counter file is SYNC_COUNT)
    counter = Path(os.environ['SYNC_COUNT'])
    n = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(n))
    if n == int(os.environ['SYNC_FAIL_ON']):
        raise SystemExit('network error (stand-in)')
if rel == 'scores/node402':
    src = Path(os.environ['NODE_FILE'])
    dst = out / 'scores' / 'node402' / 'rederive_partitions.json'
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        raise SystemExit('REFUSING: differing local files:\\nrederive_partitions.json')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
print('fetched', rel)
"""


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def world(tmp_path):
    out = tmp_path / 'outputs'
    for gate in ('n1', 'n2', 'n3', 'v1a_train', 'v1a_scorer'):
        put(out / 'gates' / f'{gate}.json', json.dumps(dict(pass_=True)).replace('pass_', 'pass'))
    put(out / 'sweep' / 'node402' / 'stage1.json',
        json.dumps(dict(outcome='GRID-4', grid=dict(new=['k1=0.1+k2=0.2', 'k1=0.1+k3=0.3', 'k2=0.2+k3=0.3', 'k4=1+k5=2']))))
    put(out / 'sweep' / 'node402' / 'stage2.json', json.dumps(dict(outcome='TOP-1')))
    put(tmp_path / 'cluster', CLUSTER)
    put(tmp_path / 'python', WRAPPER.format(python=sys.executable))
    put(tmp_path / 'sync.py', SYNC)
    for name in ('cluster', 'python'):
        (tmp_path / name).chmod(0o755)
    put(tmp_path / 'node_stage1.json', json.dumps(dict(passed=True, runs=['G'])))
    return dict(out=out, tmp=tmp_path)


STAGE1_NODE = '{"pass": true, "runs": ["stage1"]}\n'


def run_plan(world, phase, node_text=STAGE1_NODE, node_after=None, **env_extra):
    """node_text: the node's rederive_partitions.json at the start; node_after: what the vp job leaves there."""
    tmp, out = world['tmp'], world['out']
    node_file = tmp / 'node_file.json'
    put(node_file, node_text)
    after = tmp / 'node_after.json'
    put(after, node_after if node_after is not None else node_text)
    env = dict(os.environ, RS_OUTPUTS=str(out), RS_CLUSTER=str(tmp / 'cluster'), RS_PYTHON=str(tmp / 'python'),
               RS_SYNC_PY=str(tmp / 'sync.py'), NODE_FILE=str(node_file), NODE_AFTER=str(after), RS_TAG='rstest',
               RS_START_BY='2026-10-27 20:30', RS_END_BY='2026-10-28 04:30', RS_FAKE_NOW='2026-10-27 20:10')
    env.pop('RS_DRY', None)
    for key, value in env_extra.items():     # a value of None removes the variable
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run(['bash', str(PLAN), phase], env=env, capture_output=True, text=True, timeout=200)
    return proc.returncode, proc.stdout + proc.stderr


def stage1_copy(world, text=STAGE1_NODE):
    path = world['out'] / 'scores' / 'node402' / 'rederive_partitions.json'
    put(path, text)
    return path


# ---- the deadlines -----------------------------------------------------------------------------------------------

def test_stage1_runs_when_now_is_before_start_by(world):
    rc, text = run_plan(world, 'stage1')
    assert rc == 0, text
    assert 'RS_NODE_DONE stage1' in text and 'gate v1a_train holds' in text and 'gate v1a_scorer holds' in text


def test_stage1_stops_when_now_is_after_start_by(world):
    rc, text = run_plan(world, 'stage1', RS_FAKE_NOW='2026-10-27 20:31')
    assert rc == 1
    assert 'RS_NODE_STOP stage 1 cannot start on the node by RS_START_BY' in text and 'after RS_START_BY 2026-10-27 20:30' in text
    assert 'cluster launch' not in text


def test_stage1_start_deadline_is_a_date_not_a_clock_time(world):
    # 01:00 the next day is after 20:30 of the day before, though 01:00 < 20:30 as a clock time
    rc, text = run_plan(world, 'stage1', RS_FAKE_NOW='2026-10-28 01:00')
    assert rc == 1 and 'RS_NODE_STOP' in text


def test_stage1_default_start_by_is_strict_and_noted(world):
    env_free = dict(os.environ)
    env_free.pop('RS_START_BY', None)
    tmp, out = world['tmp'], world['out']
    env = dict(env_free, RS_OUTPUTS=str(out), RS_CLUSTER=str(tmp / 'cluster'), RS_PYTHON=str(tmp / 'python'),
               RS_SYNC_PY=str(tmp / 'sync.py'), RS_FAKE_NOW='2099-01-01 00:00')
    proc = subprocess.run(['bash', str(PLAN), 'stage1'], env=env, capture_output=True, text=True, timeout=200)
    text = proc.stdout + proc.stderr
    assert proc.returncode == 1 and 'RS_START_BY is not set' in text and 'RS_NODE_STOP' in text


def test_old_hhmm_form_is_refused(world):
    rc, text = run_plan(world, 'stage1', RS_START_BY='14:00')
    assert rc == 2 and 'bad start_by' in text


def test_stage2_stops_at_start_when_now_is_after_end_by(world):
    rc, text = run_plan(world, 'stage2', RS_FAKE_NOW='2026-10-28 04:31')
    assert rc == 1
    assert 'RS_NODE_STOP stage 2 has not ended by RS_END_BY' in text and 'cluster launch' not in text


def test_stage2_next_day_before_end_by_does_not_stop(world):
    # stage 1 started 20:10, so the end is 04:10 the next day; now 01:00 the next day is before it
    rc, text = run_plan(world, 'stage2', RS_START_BY='2026-10-27 20:10', RS_END_BY='2026-10-28 04:10',
                        RS_FAKE_NOW='2026-10-28 01:00')
    assert rc == 0, text
    assert 'RS_NODE_DONE stage2' in text and 'RS_NODE_STOP' not in text


def test_stage2_stops_before_rs_sweep_stage2_when_the_watch_ran_late(world, tmp_path):
    # the clock moves on during the plan: the cluster stand-in for the vp job sets the stamp file that the plan reads
    stamp = tmp_path / 'now.txt'
    stamp.write_text('2026-10-28 01:00')
    cluster = tmp_path / 'cluster'
    cluster.write_text('#!/usr/bin/env bash\necho "cluster $*"\n[[ "$*" == *rstest-stage2-vp* && "$1" == watch ]] && '
                       f'echo "2026-10-28 04:31" > {stamp}\nexit 0\n')   # no vp rewrite is needed here
    # RS_FAKE_NOW is read as a fixed string by the plan; this test needs a moving clock, so the plan's now_stamp reads a file
    # through a date shim on PATH instead
    shim = tmp_path / 'bin'
    shim.mkdir()
    (shim / 'date').write_text(f'#!/usr/bin/env bash\nif [[ "$*" == *"%F %H:%M"* ]]; then cat {stamp}; else exec /usr/bin/date "$@"; fi\n')
    (shim / 'date').chmod(0o755)
    rc, text = run_plan(world, 'stage2', RS_FAKE_NOW='', PATH=f'{shim}:{os.environ["PATH"]}')
    assert rc == 1, text
    assert 'RS_NODE_STOP stage 2 has not ended by RS_END_BY' in text
    assert text.index('launch --node node402 --tag rstest-stage2-vp') < text.index('RS_NODE_STOP')
    assert 'rs_sweep.py stage2' not in text


def test_stage1_start_by_minute_itself_does_not_stop(world):
    # under the user's exception RS_START_BY is the real start minute: a plan that starts in that minute runs
    rc, text = run_plan(world, 'stage1', RS_START_BY='2026-10-27 20:30', RS_FAKE_NOW='2026-10-27 20:30')
    assert rc == 0, text
    assert 'RS_NODE_DONE stage1' in text and 'RS_NODE_STOP' not in text
    rc, text = run_plan(world, 'stage1', RS_START_BY='2026-10-27 20:30', RS_FAKE_NOW='2026-10-27 20:31')
    assert rc == 1 and 'RS_NODE_STOP stage 1 cannot start' in text


def test_stage2_end_by_minute_itself_does_not_stop(world):
    rc, text = run_plan(world, 'stage2', RS_END_BY='2026-10-28 04:30', RS_FAKE_NOW='2026-10-28 04:30')
    assert rc == 0, text
    assert 'RS_NODE_DONE stage2' in text and 'RS_NODE_STOP' not in text
    rc, text = run_plan(world, 'stage2', RS_END_BY='2026-10-28 04:30', RS_FAKE_NOW='2026-10-28 04:31')
    assert rc == 1 and 'RS_NODE_STOP stage 2 has not ended by RS_END_BY' in text


def amsterdam_today():
    return datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F')


@pytest.mark.parametrize('clock, stops', [('14:00', False), ('14:01', True)])
def test_stage1_default_start_by_is_14_00_today(world, clock, stops):
    today = amsterdam_today()
    rc, text = run_plan(world, 'stage1', RS_START_BY=None, RS_END_BY=None, RS_FAKE_NOW=f'{today} {clock}')
    assert 'RS_START_BY is not set' in text
    if stops:
        assert rc == 1 and f'after RS_START_BY {today} 14:00' in text and 'cluster launch' not in text
    else:
        assert rc == 0, text
        assert 'RS_NODE_DONE stage1' in text


@pytest.mark.parametrize('clock, stops', [('22:00', False), ('22:01', True)])
def test_stage2_default_end_by_is_22_00_today(world, clock, stops):
    today = amsterdam_today()
    rc, text = run_plan(world, 'stage2', RS_START_BY=None, RS_END_BY=None, RS_FAKE_NOW=f'{today} {clock}')
    assert 'RS_END_BY is not set' in text
    if stops:
        assert rc == 1 and f'after RS_END_BY {today} 22:00' in text and 'cluster launch' not in text
    else:
        assert rc == 0, text
        assert 'RS_NODE_DONE stage2' in text


# ---- V1a ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('gate', ['v1a_train', 'v1a_scorer'])
def test_stage1_stops_without_v1a(world, gate):
    put(world['out'] / 'gates' / f'{gate}.json', '{"pass": false}')
    rc, text = run_plan(world, 'stage1')
    assert rc == 1 and f'RS_NODE_STOP gate {gate} does not hold' in text and 'cluster launch' not in text
    assert 'cluster sync' not in text


def test_stage1_stops_with_v1a_file_missing(world):
    (world['out'] / 'gates' / 'v1a_scorer.json').unlink()
    rc, text = run_plan(world, 'stage1')
    assert rc == 1 and 'gate v1a_scorer does not hold' in text


# ---- the move of the stage 1 partitions file ----------------------------------------------------------------------

STAGE2_NODE = json.dumps({'pass': True, 'runs': ['stage1', 'stage2']})


def test_stage2_moves_the_stage1_copy_aside_and_the_fetch_goes_through(world):
    path = stage1_copy(world)
    old = path.read_bytes()
    rc, text = run_plan(world, 'stage2', node_after=STAGE2_NODE)
    assert rc == 0, text
    moved = path.with_name('rederive_partitions.stage1.json')
    assert moved.read_bytes() == old
    assert json.loads(path.read_text())['runs'] == ['stage1', 'stage2']        # the fetch brought the node's file
    assert 'RUN move_stage1_partitions' in text


def test_stage2_without_the_move_the_fetch_refuses(world):
    # the failure the move prevents: the stand-in fetch refuses a differing local file, as das6_sync_rs.fetch does
    path = stage1_copy(world)
    node = world['tmp'] / 'n.json'
    node.write_text(STAGE2_NODE)
    proc = subprocess.run([sys.executable, str(world['tmp'] / 'sync.py'), '--fetch', 'scores/node402'],
                          env=dict(os.environ, RS_OUTPUTS=str(world['out']), NODE_FILE=str(node)), capture_output=True, text=True)
    assert proc.returncode != 0 and 'REFUSING' in proc.stderr
    assert path.exists()


def test_stage2_does_not_overwrite_a_differing_stage1_name_and_the_fetch_decides(world):
    # the stage 1 name exists with other content and a stage 1 file is at home: no move, no stop at the move; the fetch of the
    # node's stage 2 file refuses the differing home file (the stand-in applies das6_sync_rs's rule), so the plan stops there
    path = stage1_copy(world)
    other = path.with_name('rederive_partitions.stage1.json')
    other.write_text('{"pass": true, "runs": ["something else"]}\n')
    rc, text = run_plan(world, 'stage2', node_after=STAGE2_NODE)
    assert rc == 1, text
    assert 'already aside' in text and 'RUN move_stage1_partitions' not in text
    assert 'RS_NODE_STOP fetch_scores_node402 failed' in text
    assert other.read_text() == '{"pass": true, "runs": ["something else"]}\n'
    assert path.read_text() == STAGE1_NODE                       # nothing was moved or overwritten


def test_stage2_rerun_after_a_failed_second_fetch_continues(world):
    # attempt 1 moves the stage 1 copy aside, then the second fetch fails; attempt 2 starts again: its first fetch brings the
    # node's (stage 2) file home, the move step finds the stage 1 name and goes on, and the second fetch succeeds
    path = stage1_copy(world)
    old = path.read_bytes()
    moved = path.with_name('rederive_partitions.stage1.json')
    rc, text = run_plan(world, 'stage2', node_after=STAGE2_NODE, SYNC_FAIL_ON='2', SYNC_COUNT=str(world['tmp'] / 'count1'))
    assert rc == 1 and 'RUN move_stage1_partitions' in text and 'RS_NODE_STOP fetch_scores_node402 failed' in text
    assert moved.read_bytes() == old and not path.exists()
    rc, text = run_plan(world, 'stage2', node_text=STAGE2_NODE, node_after=STAGE2_NODE)
    assert rc == 0, text
    assert 'already aside' in text and 'RUN move_stage1_partitions' not in text and 'RS_NODE_DONE stage2' in text
    assert moved.read_bytes() == old                              # the stage 1 copy is untouched
    assert json.loads(path.read_text())['runs'] == ['stage1', 'stage2']


def test_stage2_rerun_after_a_good_fetch_stops_at_the_fetch_not_at_the_move(world):
    # after a good fetch the home file is the node's stage 2 file and the stage 1 name exists; a rerun whose vp job leaves a
    # different file on the node: the move step passes, the fetch refuses the differing home file
    path = stage1_copy(world, STAGE2_NODE)
    moved = path.with_name('rederive_partitions.stage1.json')
    put(moved, STAGE1_NODE)
    rc, text = run_plan(world, 'stage2', node_text=STAGE2_NODE, node_after=STAGE1_NODE)
    assert rc == 1, text
    assert 'already aside' in text and 'RS_NODE_STOP fetch_scores_node402 failed' in text
    assert 'RS_NODE_STOP' in text and text.count('RS_NODE_STOP') == 1 and 'move_stage1_partitions failed' not in text
    assert moved.read_text() == STAGE1_NODE and path.read_text() == STAGE2_NODE


def test_stage2_with_no_home_copy_yet_the_first_fetch_brings_it_and_it_moves(world):
    # the usual case after a stage 1 run on another checkout: no copy at home; the first stage 2 fetch brings the node's
    # stage 1 file (the node still holds it), the move puts it aside, the second fetch brings the new one
    rc, text = run_plan(world, 'stage2', node_after=STAGE2_NODE)
    assert rc == 0, text
    d = world['out'] / 'scores' / 'node402'
    assert json.loads((d / 'rederive_partitions.stage1.json').read_text())['runs'] == ['stage1']
    assert json.loads((d / 'rederive_partitions.json').read_text())['runs'] == ['stage1', 'stage2']


def test_stage1_has_no_move_step(world):
    stage1_copy(world)
    rc, text = run_plan(world, 'stage1')
    assert rc == 0, text
    assert 'move_stage1_partitions' not in text
    assert not (world['out'] / 'scores' / 'node402' / 'rederive_partitions.stage1.json').exists()


def test_dry_stage2_prints_the_move_before_the_second_fetch_and_stage1_does_not(world):
    base = dict(os.environ, RS_DRY='1', RS_OUTPUTS=str(world['out']), RS_START_BY='2026-10-27 20:30',
                RS_END_BY='2026-10-28 04:30', RS_FAKE_NOW='2026-10-27 20:10')
    two = subprocess.run(['bash', str(PLAN), 'stage2'], env=base, capture_output=True, text=True, timeout=200)
    one = subprocess.run(['bash', str(PLAN), 'stage1'], env=base, capture_output=True, text=True, timeout=200)
    assert two.returncode == 0 and one.returncode == 0
    lines = [ln for ln in two.stdout.splitlines() if ln.startswith('DRY')]
    names = [ln.split(':')[0] for ln in lines]
    i = names.index('DRY move_stage1_partitions')
    assert names[i - 1].startswith('DRY pull_') and names[i + 1] == 'DRY fetch_scores_node402'
    assert 'launch_vp' in ' '.join(names[:i])
    assert 'move_stage1_partitions' not in one.stdout
    assert 'gate v1a_train needs pass' in one.stdout and 'RS_START_BY' in one.stdout
    assert not (world['out'] / 'scores').exists()      # a dry run writes nothing
