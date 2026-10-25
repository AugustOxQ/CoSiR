"""Stage iii scorer: effects against R6-5 and reading a run, on small synthetic inputs."""
import json

import numpy as np
import pytest

import eg_paths
import eg_score3 as s3
import eg_variants as V
import rb_pick


def synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, size=n)
    y[rng.random(n) < 0.2] = -1                                       # masked rows
    informative = np.where(y >= 0, y, 0)

    def part(noise):
        return np.stack([np.where(rng.random(n) < noise, rng.integers(0, 5, size=n), informative)
                         for _ in range(3)]).astype(np.int64)
    return y.astype(np.int64), {'REF': part(0.6), 'GOOD': part(0.1), 'BAD': part(1.0)}


# ------------------------------------------------------------------ effects against R6-5
def test_effects_match_the_rb_function_when_the_reference_is_its_own():
    y, parts = synthetic()
    parts = {'R0-5': parts['REF'], 'GOOD': parts['GOOD'], 'BAD': parts['BAD']}
    mine = s3.aspect_effects(y, parts, 'R0-5', n_draws=40, workers=1)
    theirs = rb_pick.aspect_effects(y, parts, 40, 1)
    for n in parts:
        for k in ('A', 'e', 'lo', 'hi'):
            assert mine[n][k] == pytest.approx(theirs[n][k], abs=1e-12), (n, k)


def test_effects_use_the_named_reference():
    y, parts = synthetic()
    eff = s3.aspect_effects(y, parts, 'REF', n_draws=40, workers=1)
    assert eff['REF']['e'] == 0 and eff['REF']['lo'] == 0 and eff['REF']['hi'] == 0
    assert eff['GOOD']['e'] == pytest.approx(eff['GOOD']['A'] - eff['REF']['A'])
    assert eff['GOOD']['e'] > 0 > eff['BAD']['e']
    assert eff['GOOD']['lo'] > 0 > eff['BAD']['hi']


def test_cached_bootstrap_scores_are_reused_and_arms_are_independent(monkeypatch):
    y, parts = synthetic()
    store = {}

    class Cache:
        def get(self, n):
            return store.get(n)

        def put(self, n, b):
            store[n] = b
    first = s3.aspect_effects(y, parts, 'REF', n_draws=30, workers=1, cache=Cache())
    assert set(store) == set(parts)
    with monkeypatch.context() as m:
        m.setattr(rb_pick, 'bootstrap_scores', lambda *a, **k: pytest.fail('bootstrap recomputed'))
        assert s3.aspect_effects(y, parts, 'REF', n_draws=30, workers=1, cache=Cache()) == first
    del store['GOOD']                  # one arm computed alone gives the numbers it had when computed with the others
    assert s3.aspect_effects(y, parts, 'REF', n_draws=30, workers=1, cache=Cache()) == first


# ------------------------------------------------------------------ reading a run (VG3, failed runs, VG4 folders)
PSHA = eg_paths.P_SHA256


def write_run(out, variant, status='ok', cause=None, n=29232, norm=1.0, finite=True, p_sha=PSHA):
    folder = eg_paths.iii_run(out, variant, 0)
    folder.mkdir(parents=True)
    run = dict(variant=variant, seed=0, switches=V.switches(variant), status=status, cause=cause,
               best_epoch={a: 3 for a in eg_paths.ASPECTS}, p_sha256=p_sha)
    if status == 'failed':
        run['failure_kind'] = 'nonfinite_loss'
    if p_sha is None:
        del run['p_sha256']
    (folder / 'run.json').write_text(json.dumps(run))
    if status == 'ok':
        x = np.zeros((n, 2048), np.float16)
        x[:, 0] = norm
        if not finite:
            x[0, 0] = np.nan
        for a in eg_paths.ASPECTS:
            np.savez(folder / f'{a}__P.npz', img=x)


def test_read_run_ok_failed_declared_and_vg3(tmp_path):
    write_run(tmp_path, 'GT-2')
    r = s3.read_run(tmp_path, 'GT-2', PSHA)
    assert r['status'] == 'ok' and set(r['img']) == {'emotion', 'style', 'genre'} and r['best_epoch']['genre'] == 3
    write_run(tmp_path, 'GT-4', status='failed', cause='non-finite loss at epoch 2')
    r = s3.read_run(tmp_path, 'GT-4', PSHA)
    assert r['status'] == 'failed' and r['img'] is None and 'non-finite' in r['cause']
    write_run(tmp_path, 'GR-2', norm=0.5)                              # fails VG3 (not unit norm)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GR-2', PSHA)
    r = s3.read_run(tmp_path, 'GR-2', PSHA, {'GR-2': 'norm 0.5 after the rerun'})
    assert r['status'] == 'failed' and r['cause'] == 'norm 0.5 after the rerun'
    assert set(r['vg3_problem']) == {'emotion', 'style', 'genre'} and r['img'] is None     # the problem beside the cause
    write_run(tmp_path, 'GA-1', n=100)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GA-1', PSHA)
    write_run(tmp_path, 'GR-0.5', finite=False)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GR-0.5', PSHA)


def test_declared_failure_of_a_run_that_passes_vg3_is_refused(tmp_path):
    write_run(tmp_path, 'GT-2')                                        # exports pass VG3 on all three aspects
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GT-2', PSHA, {'GT-2': 'no good reason'})


def test_declared_failure_needs_one_aspect_failing_vg3(tmp_path):
    write_run(tmp_path, 'GT-2')
    folder = eg_paths.iii_run(tmp_path, 'GT-2', 0)
    bad = np.zeros((29232, 2048), np.float16)                          # genre alone is not unit norm
    np.savez(folder / 'genre__P.npz', img=bad)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GT-2', PSHA)
    r = s3.read_run(tmp_path, 'GT-2', PSHA, {'GT-2': 'genre norm 0 after the rerun'})
    assert r['status'] == 'failed' and list(r['vg3_problem']) == ['genre'] and 'norm 0' in r['cause']


def test_run_json_must_carry_the_p_hash_of_12_24(tmp_path):
    write_run(tmp_path, 'GT-2', p_sha=None)
    with pytest.raises(AssertionError, match='p_sha256'):
        s3.read_run(tmp_path, 'GT-2', PSHA)
    write_run(tmp_path, 'GT-4', status='failed', cause='non-finite loss', p_sha=None)
    with pytest.raises(AssertionError, match='p_sha256'):
        s3.read_run(tmp_path, 'GT-4', PSHA)
    write_run(tmp_path, 'GR-2', p_sha='0' * 64)
    with pytest.raises(AssertionError, match='p_sha256'):
        s3.read_run(tmp_path, 'GR-2', PSHA)


def test_failed_option_refuses_a_variant_outside_the_scored_set():
    first = list(V.FIRST)
    assert s3.parse_failed(['GT-2=bad norm'], first) == {'GT-2': 'bad norm'}
    with pytest.raises(AssertionError, match='scored'):
        s3.parse_failed(['GT-6=bad norm'], first)                      # a valid variant, but not run in this stage
    assert s3.parse_failed(['GT-6=bad norm'], first + ['GT-6']) == {'GT-6': 'bad norm'}


def test_scorer_refuses_a_declared_variant_outside_the_variants_it_scores(tmp_path):
    with pytest.raises(AssertionError, match='scored'):
        s3.score_variants(tmp_path, ['GT-2'], 20, 1, {'GT-4': 'x'}, ctx=dict(p_sha=PSHA, labels={}))


def test_bootstrap_cache_identity_names_the_code_it_runs():
    ident = s3._identity('GT-2', 'genre', 'p', 'i')
    assert set(ident['code_sha256']) == {'rb_ami.py', 'rb_pick.py'}


def test_variant_folders_outside_the_allowed_set_stop_the_scorer(tmp_path):
    write_run(tmp_path, 'GT-2')
    write_run(tmp_path, 'R6-5')
    s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4'])
    write_run(tmp_path, 'GT-6')
    with pytest.raises(AssertionError, match='VG4'):
        s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4'])
    s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4', 'GT-6'])


@pytest.mark.parametrize('old_hash', ['old scorer', None])
def test_existing_score_refuses_stale_code_without_overwriting(tmp_path, old_hash):
    path = tmp_path / 'score_first.json'
    old = dict(identity={'p_sha256': 'p'}, code_sha256=old_hash)
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    record = dict(identity=old['identity'], **s3.code_state())
    with pytest.raises(AssertionError, match=f'move {path} aside, then rerun'):
        s3._write_or_skip(path, record)
    assert path.read_bytes() == before


def test_score_code_hash_covers_imported_round_modules(monkeypatch):
    original = s3._file_sha256
    before = s3.code_state()['code_sha256']
    monkeypatch.setattr(s3, '_file_sha256', lambda p: 'changed' if p.name == 'eg_variants.py' else original(p))
    assert s3.code_state()['code_sha256'] != before


@pytest.mark.parametrize('module, changes_hash', [
    ('eg_pinfer.py', False), ('eg_score.py', False), ('eg_run.py', False),
    ('eg_score3.py', True), ('eg_paths.py', True), ('eg_variants.py', True),
    ('rb_ami.py', True), ('rb_data.py', True), ('rb_pick.py', True),
    ('rb_paths.py', True), ('rb_rows.py', True), ('rb_score3_dependency.py', True),
])
def test_score_code_hash_tracks_only_static_module_dependencies(tmp_path, monkeypatch, module, changes_hash):
    import shutil
    from pathlib import Path

    folders = []
    for source, pattern in ((eg_paths.ROUND, 'eg_*.py'), (Path(s3.rb_ami.__file__).parent, 'rb_*.py')):
        folder = tmp_path / source.name
        folder.mkdir()
        for path in source.glob(pattern):
            shutil.copyfile(path, folder / path.name)
        folders.append(folder)
    round_folder, rb_folder = folders
    # A from-import reached through rb_pick -> rb_rows; never imported at runtime.
    rows = rb_folder / 'rb_rows.py'
    rows.write_text(rows.read_text() + '\nfrom rb_score3_dependency import value as dependency_value\n')
    (rb_folder / 'rb_score3_dependency.py').write_text('import rb_rows\nvalue = 1\n')  # cycle
    monkeypatch.setattr(eg_paths, 'ROUND', round_folder)
    monkeypatch.setattr(s3.rb_ami, '__file__', str(rb_folder / 'rb_ami.py'))
    before = s3.scorer_code_sha256()
    # A score remains valid after unrelated edits, and is refused after dependency edits.
    score = tmp_path / 'score_first.json'
    score.write_text(json.dumps(dict(code_sha256=before)))
    path = (round_folder if module.startswith('eg_') else rb_folder) / module
    path.write_text(path.read_text() + '\n# changed after scoring\n')
    assert (s3.scorer_code_sha256() != before) == changes_hash
    if changes_hash:
        with pytest.raises(AssertionError, match='stale code_sha256'):
            s3.check_score_code(score)
    else:
        s3.check_score_code(score)


@pytest.mark.parametrize('phase', ['first', 'all', 'fresh'])
def test_queue_refuses_stale_score_before_any_work(tmp_path, phase):
    import os
    import subprocess
    import sys
    stage = tmp_path / 'stage_iii'
    stage.mkdir()
    path = stage / f'score_{phase}.json'
    path.write_text(json.dumps(dict(code_sha256='old scorer', identity={})))
    # Any work other than the real code preflight fails immediately; no runner sees real outputs.
    wrapper = tmp_path / 'python'
    wrapper.write_text(f'#!{sys.executable}\nimport os, sys\n'
                       'if "check-code" in sys.argv:\n'
                       '    os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n'
                       'print("UNEXPECTED_WORK", sys.argv)\nsys.exit(1)\n')
    wrapper.chmod(0o755)
    env = dict(os.environ, EG_OUTPUTS=str(tmp_path), EG_PYTHON=str(wrapper), EG_DEVICE='cpu',
               EG_PENDING=str(tmp_path/'pending.md'), EG_GPU_LOCK=str(tmp_path/'gpu.lock'), EG_DRY='0', EG_SMOKE='0')
    result = subprocess.run(['bash', str(eg_paths.ROUND/'eg_queue_stage_iii.sh')],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert f'move {path} aside, then rerun' in result.stdout + result.stderr
    assert 'UNEXPECTED_WORK' not in result.stdout + result.stderr


def doc_shell_block(marker):
    import re
    text = (eg_paths.ROUND/'eg_queue.md').read_text()
    blocks = re.findall(r'```bash\n(.*?)```', text, re.S)
    matching = [block for block in blocks if block.startswith(marker + '\n')]
    assert len(matching) == 1, f'missing executable documentation: {marker}'
    return matching[0]


def test_documented_stage_iii_recovery_preserves_files_and_reruns(tmp_path):
    import os
    import subprocess
    block = doc_shell_block('# stage iii recovery')
    stage = tmp_path/'stage_iii'
    stage.mkdir()
    names = [f'{kind}_{phase}.json' for kind in ('score', 'rederive_vg2') for phase in ('first', 'all', 'fresh')]
    names += ['reported.json', 'reported.md', 'vg1_scorer.json', 'partitions']
    for name in names:
        (stage/name).write_text(name)
    (stage/'train').mkdir()
    (stage/'train/keep').write_text('trained weights')
    (stage/'fresh_parity.json').write_text('parity evidence')
    commands = tmp_path/'commands'
    commands.mkdir()
    for name in ('python', 'bash'):
        path = commands/name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "' + name + ' $*" >> "$CALLS"\n')
        path.chmod(0o755)
    calls = tmp_path/'calls'
    env = dict(os.environ, EG_OUTPUTS=str(tmp_path), EG_PYTHON=str(commands/'python'),
               PATH=str(commands)+os.pathsep+os.environ['PATH'], CALLS=str(calls))
    subprocess.run(['/bin/bash', '-euc', block], env=env, cwd=eg_paths.ROUND, check=True, timeout=10)
    for name in names:
        moved = list((stage/'aside').glob(f'*/{name}'))
        assert len(moved) == 1 and moved[0].read_text() == name
        assert not (stage/name).exists()
    assert (stage/'train/keep').read_text() == 'trained weights'
    assert (stage/'fresh_parity.json').read_text() == 'parity evidence'
    assert calls.read_text().splitlines() == [
        f'python eg_score3.py check vg1 --outputs {tmp_path}', 'bash eg_queue_stage_iii.sh']


def test_documented_fresh_failed_flags_pass_both_real_cli_parsers(tmp_path, monkeypatch):
    import shlex
    import eg_vg2
    block = doc_shell_block('# fresh VG3 failure')
    stage = tmp_path/'stage_iii'
    stage.mkdir()
    (stage/'score_all.json').write_text(json.dumps(dict(winner='GT-4')))
    captured = {}
    def score(outputs, draws, workers, smoke, failed):
        captured['scorer'] = failed
        return dict(winner='GT-4', c_outcome=None, no_outcome_cause='VG3', reading='no C outcome')
    def verify(*args, **kwargs):
        captured['vg2'] = kwargs['failed']
        return dict(agree=True, differences=[], n_draws=20, complete=True)
    monkeypatch.setattr(s3, 'run_fresh', score)
    monkeypatch.setattr(eg_vg2, 'run', verify)
    for line in block.splitlines():
        if line.startswith('python '):
            args = [word.replace('$EG_OUTPUTS', str(tmp_path)) for word in shlex.split(line)]
            main = s3.main if args[1] == 'eg_score3.py' else eg_vg2.main
            assert main(args[2:]) == 0
    assert captured == dict(scorer={'GT-4': 'VG3 after forced rerun'},
                            vg2={'GT-4/s2': 'VG3 after forced rerun'})
