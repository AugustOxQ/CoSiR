"""The full-scale trainer and held inference with the knobs (ticket 08): the provenance check, the guards, the knob
layer reaching the full path (patched objects, tiny CPU tensors), the run logic with the cohort replaced by a
recorder, and V4a's comparison. The cohort itself is `rg_full.train_cohort` (`ct2_d2.run`), checked by V4a on the GPU."""
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as Fn

import rs_full as F
import rs_knobs as K
import rs_paths as paths
import rs_sheet as S
import ab_train
import rb_encoders
import rb_rows
import rb_student
import rg_full
import rg_loss
import rg_variants as V
from ab_paths import ms_train

CPU = torch.device('cpu')
ASPECTS = F.ASPECTS


def scratch(name):
    """An empty folder of this test's own under the smoke root (a smoke may write nowhere else)."""
    folder = paths.SMOKE_ROOT / name
    shutil.rmtree(folder, ignore_errors=True)
    return folder


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


# ------------------------------------------------------------------------------------------- provenance (frozen code)
def test_frozen_modules_are_the_ms_and_ct2_modules_imported_and_rg_loss_and_rg_full():
    modules = F.frozen_modules()
    assert {'ms_train', 'ms_rows', 'ct2_d2', 'ct2_data', 'ct2_graphs', 'rg_loss', 'rg_full'} <= set(modules)
    assert all(name.startswith(('ms_', 'ct2_')) or name in ('rg_loss', 'rg_full') for name in modules)
    for name, record in modules.items():
        assert Path(record['path']).is_file() and len(record['sha256']) == 64, name


def test_frozen_code_equals_the_hashes_at_the_frozen_commit():
    # the real check: git show of every module at 68e8081
    shas = F.require_frozen_code()
    assert shas == {name: record['sha256'] for name, record in F.frozen_modules().items()}
    assert F.paths.FROZEN_COMMIT == '68e808167fda05350dfc510ea38c627f8a0afc89'


def test_frozen_code_refuses_a_module_that_differs_from_the_commit():
    def show(commit, relative):
        return b'not the frozen file' if relative.endswith('ms_train.py') else subprocess.run(
            ['git', 'show', f'{commit}:{relative}'], capture_output=True, check=True, cwd=paths.ROUND).stdout
    with pytest.raises(AssertionError, match='ms_train.*differs from commit 68e8081'):
        F.require_frozen_code(show=show)


def test_frozen_code_refuses_a_module_the_commit_does_not_have():
    def show(commit, relative):
        raise subprocess.CalledProcessError(128, 'git show')
    with pytest.raises(AssertionError, match='not in commit'):
        F.require_frozen_code(show=show)


# ------------------------------------------------------------------------------------ the knob layer at full scale
def full_state():
    """Every object the layer may patch on the full path, by value or identity."""
    return dict(config=dict(ms_train.CONFIG), arm=dict(V.ARMS['GS-0.02']), weight=V.TEACHER_WEIGHT,
                switched=ab_train.switched_head_loss, recipe=rb_rows.recipe, load_scale=rb_student.load_scale,
                save_output=rb_student.CT2_SAVE_OUTPUT, infer=rb_student.infer, trunk=ms_train.TrunkSide)


def touched(cfg):
    before = full_state()
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='full', outputs='o')
        inside = full_state()
    assert full_state() == before
    return {name for name in before if inside[name] != before[name]}


def test_g_enters_no_patch_at_full_scale():
    stack = ExitStack()
    assert K.patches(S.config(), stack, scale='full') == {}
    assert not stack._exit_callbacks
    assert touched(S.config()) == set()


@pytest.mark.parametrize('knob,parts', [
    ('k1', {'switched'}), ('k2', {'switched'}), ('k3', {'arm'}), ('k4', {'arm'}), ('k5', {'recipe', 'weight'}),
    ('k6', {'config'}), ('k7', {'config'}), ('k8', {'config', 'save_output', 'infer'}),
    ('k9', {'config', 'trunk'}), ('k10', {'config', 'trunk'}), ('k11', {'load_scale'}), ('k12', {'config'}),
    ('k13', {'config'})])
def test_each_knob_patches_only_its_part_at_full_scale(knob, parts):
    for value in S.VALUES[knob]:
        cfg = S.config({knob: value})
        assert touched(cfg) == parts
        assert set(K.describe(cfg, scale='full')) == {knob}


def test_the_sample_description_is_unchanged():
    assert K.describe(S.config({'k11': 30}))['k11'] == {'rg_train.load_graphs': "rs_graphs.sample_store(k=30, 'aspect')"}
    assert K.describe(S.config({'k5': 0.5}))['k5']['rg_train.resolved'] == 'direct weight 0.5'
    assert K.describe(S.config({'k5': 0.5}), scale='full')['k5']['rb_rows.recipe'] == 'direct weight 0.5'


def tiny(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    img, cap, teacher = (Fn.normalize(torch.randn(2 * n, 2048, generator=g), dim=1) for _ in range(3))
    pairs = torch.stack((torch.arange(n), torch.arange(n, 2 * n)), dim=1).numpy()
    return img, cap, teacher, pairs, torch.ones(n).numpy()


def full_head_loss(cfg, aspect):
    """The loss the full path computes for one head under `cfg`: `ms_train.head_loss` as `rg_full.train_cohort` patches
    it (`head_loss_for`), called as `ct2_d2.run` calls it (tensors by keyword), recipe row from `rb_rows.recipe`."""
    img, cap, teacher, pairs, weights = tiny()
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='full')
        row = rb_rows.recipe(V.BASE)['base']
        loss_fn = rg_full.head_loss_for('GS-0.02', 0, fit_teacher=None, n_dev=1)
        torch.manual_seed(3)
        models, views = ms_train.build_model(row, list(ASPECTS), CPU)
        models.eval()
        loss = loss_fn(views[aspect], img=img, cap=cap, teacher=teacher, pairs=pairs, weights=weights, row=row,
                       aspect=aspect)
        z = views[aspect]['img'](img[torch.as_tensor(pairs).T.reshape(-1)])
    return float(loss.detach()), z, teacher, pairs


def relational_of(z, teacher, pairs, tau_t):
    ids = torch.as_tensor(pairs).T.reshape(-1)
    return float(rg_loss.relational_loss(z, teacher[ids], ids, tau_t).detach())


@pytest.mark.parametrize('knob,aspect,other', [('k1', 'emotion', 'style'), ('k2', 'style', 'emotion')])
def test_k1_k2_reach_the_full_loss(knob, aspect, other):
    base, z, teacher, pairs = full_head_loss(S.config(), aspect)
    for weight in (2, 16):
        loss, _, _, _ = full_head_loss(S.config({knob: weight}), aspect)
        assert loss == pytest.approx(base + weight * relational_of(z, teacher, pairs, 0.02), rel=1e-5)
        assert full_head_loss(S.config({knob: weight}), other)[0] == full_head_loss(S.config(), other)[0]


def test_k3_k4_reach_the_full_genre_loss():
    base, z, teacher, pairs = full_head_loss(S.config(), 'genre')
    loss, _, _, _ = full_head_loss(S.config({'k3': 12}), 'genre')
    assert loss == pytest.approx(base + 4 * relational_of(z, teacher, pairs, 0.02), rel=1e-5)
    loss, _, _, _ = full_head_loss(S.config({'k4': 0.035}), 'genre')
    assert loss == pytest.approx(base + 8 * (relational_of(z, teacher, pairs, 0.035)
                                            - relational_of(z, teacher, pairs, 0.02)), rel=1e-5)


def test_k5_changes_the_training_recipe_row_only():
    frozen = rb_rows.recipe(V.BASE)
    with ExitStack() as stack:
        K.patches(S.config({'k5': 0.5}), stack, scale='full')
        row = rb_rows.recipe(V.BASE)
        assert row['base']['distill_weight'] == 0.5 and V.TEACHER_WEIGHT == 0.5
        assert row['switches'] == frozen['switches'] and row['store'] == frozen['store']
        assert {k: v for k, v in row['base'].items() if k != 'distill_weight'} == \
            {k: v for k, v in frozen['base'].items() if k != 'distill_weight'}
        assert rb_rows.recipe(rb_rows.REFERENCE) == rb_rows.RECIPES[rb_rows.REFERENCE]    # inference's recipe: untouched
    assert rb_rows.recipe(V.BASE)['base']['distill_weight'] == 0.25 and V.TEACHER_WEIGHT == 0.25


def test_k8_pins_the_full_exports_and_held_inference_but_not_training(monkeypatch):
    seen = []
    monkeypatch.setattr(rb_student, 'infer', lambda *a, **k: seen.append(('infer', ms_train.CONFIG['batch_size'])))
    monkeypatch.setattr(rb_student, 'CT2_SAVE_OUTPUT', lambda *a, **k: seen.append(('save', ms_train.CONFIG['batch_size'])))
    for cfg, training_batch in ((S.config(), 512), (S.config({'k8': 2048}), 2048)):
        seen.clear()
        with ExitStack() as stack:
            K.patches(cfg, stack, scale='full')
            assert ms_train.CONFIG['batch_size'] == training_batch
            rb_student.infer('w', 'genre', 2048, None, CPU)
            rb_student.CT2_SAVE_OUTPUT()
            assert ms_train.CONFIG['batch_size'] == training_batch
        assert seen == [('infer', 512), ('save', 512)]


@pytest.mark.parametrize('width,dropout', [(512, 0.1), (2048, 0.1), (1024, 0.0), (512, 0.3)])
def test_k9_k10_build_the_full_models_trunk(width, dropout):
    row = rb_rows.recipe(V.BASE)['base']
    with ExitStack() as stack:
        K.patches(S.config({'k9': width, 'k10': dropout}), stack, scale='full')
        models, views = rb_student.build_model(row, list(ASPECTS), CPU, width=2048)     # the full path's factory
        for side in ('img', 'cap'):
            assert [tuple(p.shape) for p in models[side].trunk.parameters()] == [(width, 2048), (width,)]
            assert models[side].trunk[2].p == dropout
        assert views['genre']['img'](torch.randn(2, 2048)).shape == (2, 2048)


def test_k11_replaces_the_full_loader_and_k20_never_does(monkeypatch, tmp_path):
    seen = []

    class Fake:
        @staticmethod
        def full_training_data(k, variant='aspect', *, outputs=None):
            seen.append((k, variant, outputs))
            return 's', 'inputs', 'graphs', 'manifests'

    monkeypatch.setitem(sys.modules, 'rs_graphs', Fake)
    frozen = rb_student.load_scale
    with ExitStack() as stack:
        K.patches(S.config({'k11': 30}), stack, scale='full', outputs=tmp_path)
        assert rb_student.load_scale('full') == ('s', 'inputs', 'graphs', 'manifests')
    assert seen == [(30, 'aspect', tmp_path)] and rb_student.load_scale is frozen
    with pytest.raises(AssertionError, match='needs the outputs'):
        K.patches(S.config({'k11': 30}), ExitStack(), scale='full')
    with pytest.raises(AssertionError, match='frozen'):
        K.store_training_data('full', 20, tmp_path)


def test_removals_still_wait_for_ticket_07_at_full_scale():
    with pytest.raises(AssertionError, match='ticket 07'):
        K.patches(S.config(remove=['buddy']), ExitStack(), scale='full')


# ------------------------------------------------------------------------------------------------- the run logic
VERDICT_BAD = 'stage3/verdict.json'


def put_verdict(outputs, outcome='PICK', c_star='k6=0.0005'):
    put(Path(outputs) / VERDICT_BAD, dict(outcome=outcome, c_star=c_star if outcome == 'PICK' else None))


class Run:
    """Replaces the data loader and the cohort; `calls` records what the cohort saw (the patches in force)."""

    def __init__(self, monkeypatch):
        self.calls, self.outcome, self.loads = [], 'ok', 0
        monkeypatch.setattr(F, 'load_data', self.load)
        monkeypatch.setattr(rg_full, 'train_cohort', self.cohort)

    def load(self):
        self.loads += 1
        return {'fit': np.arange(10_000)}, {a: {} for a in ASPECTS}, {}, {}, np.zeros((4, 2048), np.float32)

    def cohort(self, s, inputs, graphs, manifests, images, arm, seed, device, root, *, rb, smoke=False,
               max_epochs=None, stage_b_verdict_sha256=None):
        self.calls.append(dict(arm=arm, seed=seed, root=Path(root), lr=ms_train.CONFIG['lr'],
                               arms=dict(V.ARMS[arm]), alias=arm in V.ARMS, weight=rb_rows.recipe(V.BASE)['base']['distill_weight'],
                               smoke=smoke, max_epochs=max_epochs, stage_b=stage_b_verdict_sha256, device=str(device)))
        if self.outcome == 'nonfinite':
            record = dict(arm=arm, seed=seed, status='failed', failure_kind='nonfinite_loss', cause='non-finite loss')
            put(rg_full.failed_path(root, arm, seed), record)
            return record
        rg = rg_full.rg_block(arm, seed, len(s['fit']))
        for aspect in ASPECTS:
            path = rb_student.weights_path(root, arm, aspect, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'{arm}{aspect}{seed}{ms_train.CONFIG["lr"]}'.encode())
            config = dict(aspect=aspect, seed=seed, smoke=smoke, device=str(device), max_epochs=max_epochs or 200,
                          rb=rb, rg=rg)
            put(path.with_suffix('.json'), dict(weights_sha256=ms_train.head.sha_file(path), config=config,
                                                best_epoch=3, best_dev_loss=0.5))
        return {a: {'best_epoch': 3} for a in ASPECTS}


@pytest.fixture
def run(monkeypatch):
    return Run(monkeypatch)


def train(name='G-v4a', seeds=(0,), outputs=None, **kwargs):
    return F.train(name, seeds, 'cpu', outputs, **kwargs)


def test_g_trains_with_no_patch_under_its_run_name_and_writes_the_record(run, tmp_path):
    before = (dict(ms_train.CONFIG), dict(V.ARMS['GS-0.02']), list(V.ARMS))
    train('G-v4a', outputs=tmp_path, root=tmp_path / 'regression/v4a_run')
    [call] = run.calls
    assert (call['arm'], call['seed'], call['lr'], call['weight']) == ('G-v4a', 0, 1e-3, 0.25)
    assert call['arms'] == V.ARMS['GS-0.02'] and call['alias'] and call['root'] == (tmp_path / 'regression/v4a_run').resolve()
    assert (dict(ms_train.CONFIG), dict(V.ARMS['GS-0.02']), list(V.ARMS)) == before          # alias gone after the run
    record = json.loads((tmp_path / 'regression/v4a_run/train/G-v4a/run__s0.json').read_text())
    assert (record['name'], record['seed'], record['status'], record['config']) == (
        'G-v4a', 0, 'ok', S.config() | {'remove': []})
    assert record['knobs'] == {} and record['stage3_verdict_sha256'] is None
    assert record['frozen_commit'] == paths.FROZEN_COMMIT and record['frozen_sha256'] == F.require_frozen_code()
    assert set(record['code_sha256']) >= {'rs_full.py', 'rs_knobs.py', 'rg_full.py', 'ms_train.py'}
    assert set(record['weights_sha256']) == set(ASPECTS) and record['best_epoch'] == dict.fromkeys(ASPECTS, 3)
    assert record['graphs'] == dict(k=20, variant='aspect')


def test_a_candidate_trains_under_its_knobs_and_its_arm_carries_them(run, tmp_path):
    put_verdict(tmp_path)
    cfg = S.config({'k6': 5e-4, 'k4': 0.025})
    name = S.name(cfg)
    assert name == 'k4=0.025+k6=0.0005'
    put_verdict(tmp_path, c_star=name)
    train(name, outputs=tmp_path)
    [call] = run.calls
    assert call['arm'] == name and call['lr'] == 5e-4 and call['arms']['tau_t'] == 0.025 and call['arms']['lam'] == 8
    assert call['root'] == paths.full_root(tmp_path).resolve()
    assert name not in V.ARMS and V.ARMS['GS-0.02']['tau_t'] == 0.02 and ms_train.CONFIG['lr'] == 1e-3
    record = json.loads((paths.full_root(tmp_path) / 'train' / name / 'run__s0.json').read_text())
    assert record['knobs'] == K.describe(cfg, scale='full') and set(record['knobs']) == {'k4', 'k6'}
    assert record['stage3_verdict_sha256'] == ms_train.head.sha_file(tmp_path / VERDICT_BAD)
    assert call['stage_b'] is None


def test_a_finished_seed_is_skipped_and_other_code_blocks_it(run, tmp_path, monkeypatch):
    train(outputs=tmp_path, root=tmp_path / 'r')
    train(outputs=tmp_path, root=tmp_path / 'r')
    assert len(run.calls) == 1
    record_path = tmp_path / 'r/train/G-v4a/run__s0.json'
    record = json.loads(record_path.read_text())
    record['code_sha256']['rs_knobs.py'] = '0'
    put(record_path, record)
    with pytest.raises(AssertionError, match='code_sha256 changed'):
        train(outputs=tmp_path, root=tmp_path / 'r')
    assert len(run.calls) == 1


def test_a_changed_frozen_module_refuses_before_any_data_is_read(run, tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'require_frozen_code', lambda **k: (_ for _ in ()).throw(AssertionError('differs from commit')))
    with pytest.raises(AssertionError, match='differs from commit'):
        train(outputs=tmp_path, root=tmp_path / 'r')
    assert run.loads == 0 and not run.calls


def test_training_refuses_once_held_outputs_exist(run, tmp_path):
    root = tmp_path / 'r'
    (root / 'held/G-v4a').mkdir(parents=True)
    (root / 'held/G-v4a/genre__s3.npz').write_bytes(b'x')
    with pytest.raises(AssertionError, match='no new training'):
        train(outputs=tmp_path, root=root)
    assert run.loads == 0


def test_a_non_finite_loss_is_recorded_and_never_rerun(run, tmp_path):
    run.outcome = 'nonfinite'
    result = train(outputs=tmp_path, root=tmp_path / 'r')
    assert result[0]['status'] == 'failed'
    record = json.loads((tmp_path / 'r/train/G-v4a/run__s0.json').read_text())
    assert record['status'] == 'failed' and record['cause'] == 'non-finite loss'
    with pytest.raises(AssertionError, match='no rerun'):
        train(outputs=tmp_path, root=tmp_path / 'r')
    assert len(run.calls) == 1


@pytest.mark.parametrize('verdict,match', [
    (None, 'stage3/verdict.json'),
    (dict(outcome='KEEP-G', c_star=None), 'KEEP-G'),
    (dict(outcome='PICK', c_star='k7=0.01'), 'names k7=0.01'),
])
def test_a_stage_4_candidate_trains_only_when_the_stage_3_verdict_names_it(run, tmp_path, verdict, match):
    if verdict is not None:
        put(tmp_path / VERDICT_BAD, verdict)
    with pytest.raises(AssertionError, match=match):
        train('k6=0.0005', outputs=tmp_path)
    assert run.loads == 0


def test_run_names_must_be_a_sheet_name_or_the_regression(run, tmp_path):
    with pytest.raises(AssertionError, match='name'):
        train('G-repeat', outputs=tmp_path)
    with pytest.raises(AssertionError, match='name'):
        train('G', outputs=tmp_path)          # G itself is 12-27's frozen run: only the regression retrains it
    with pytest.raises(AssertionError, match='seed'):
        train('G-v4a', seeds=(5,), outputs=tmp_path, root=tmp_path / 'r')
    with pytest.raises(AssertionError, match='regression root'):
        train('G-v4a', outputs=tmp_path)
    assert run.loads == 0


def test_a_smoke_runs_without_the_verdict_and_writes_under_scratch_only(run):
    out = scratch('r10-rs-08-test')
    train('k6=0.0005', outputs=out, smoke=True)
    [call] = run.calls
    assert call['smoke'] and call['max_epochs'] is None and call['lr'] == 5e-4
    with pytest.raises(AssertionError, match='tmp/claude-0'):
        train('k6=0.0005', outputs=paths.OUTPUTS, smoke=True)
    with pytest.raises(AssertionError, match='at most 2 epochs'):
        train('k6=0.0005', outputs=out, smoke=True, max_epochs=3)
    with pytest.raises(AssertionError, match='frozen'):
        train('G-v4a', outputs=paths.OUTPUTS, root=paths.RG_FULL, smoke=False)


# ------------------------------------------------------------------------------------------------ held inference
def put_gates(outputs, **over):
    put(Path(outputs) / 'gates/v4a_train.json', {'pass': True, 'smoke': False, 'code_sha256': F.code_sha256(), **over})
    put(Path(outputs) / 'gates/v4a_scorer.json', {'pass': True})


class Held:
    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(rg_full, 'infer_held', self.infer)

    def infer(self, arm, seeds, device, root, *, smoke=False):
        self.calls.append(dict(arm=arm, seeds=list(seeds), root=Path(root), arms=dict(V.ARMS[arm]), lr=ms_train.CONFIG['lr']))
        for seed in seeds:
            for aspect in ASPECTS:
                put(Path(root) / 'held' / arm / f'{aspect}__s{seed}.json',
                    dict(rows=paths.N_HELD, list_sha256=paths.HELD_SHA256, sha256='x' * 64, vc3_attempts=1, vc3_problems=[],
                         arm=arm, aspect=aspect, seed=seed))
        return {}


@pytest.fixture
def held_setup(run, tmp_path, monkeypatch):
    held = Held(monkeypatch)
    name = 'k6=0.0005'
    put_verdict(tmp_path, c_star=name)
    put_gates(tmp_path)
    train(name, seeds=S.FULL_SEEDS, outputs=tmp_path)
    return held, name, tmp_path


def held(name, outputs, seeds=S.FULL_SEEDS, **kwargs):
    return F.held(name, seeds, outputs, 'cpu', ancestor=lambda: True, **kwargs)


def test_held_runs_the_five_seeds_under_the_knobs_and_records_v4c(held_setup):
    recorder, name, out = held_setup
    held(name, out)
    [call] = recorder.calls
    assert call['arm'] == name and call['seeds'] == [0, 1, 2, 3, 4] and call['lr'] == 5e-4
    assert call['root'] == paths.full_root(out).resolve() and name not in V.ARMS
    v4c = json.loads((paths.full_root(out) / 'held' / name / 'v4c.json').read_text())
    assert v4c['pass'] is True and v4c['list_sha256'] == paths.HELD_SHA256 and len(v4c['outputs']) == 15
    assert v4c['rows'] == paths.N_HELD


@pytest.mark.parametrize('gate,change,match', [
    ('v4a_train', 'delete', 'v4a_train.json'), ('v4a_scorer', 'delete', 'v4a_scorer.json'),
    ('v4a_train', dict(pass_=False), 'does not pass'), ('v4a_scorer', dict(pass_=False), 'does not pass'),
    ('v4a_train', dict(smoke=True), 'smoke'), ('v4a_train', dict(code_sha256={'rs_full.py': '0'}), 'other code')])
def test_held_refuses_without_the_v4a_gates(held_setup, gate, change, match):
    recorder, name, out = held_setup
    path = out / f'gates/{gate}.json'
    if change == 'delete':
        path.unlink()
    else:
        record = json.loads(path.read_text())
        record.update({'pass': change['pass_']} if 'pass_' in change else change)
        put(path, record)
    with pytest.raises(AssertionError, match=match):
        held(name, out)
    assert not recorder.calls and not (paths.full_root(out) / 'held').exists()


def test_held_refuses_g_and_a_candidate_the_stage_3_verdict_does_not_name(held_setup):
    recorder, name, out = held_setup
    with pytest.raises(AssertionError, match='12-27'):
        held('G', out)
    with pytest.raises(AssertionError, match='G-v4a'):
        held('G-v4a', out)
    put_verdict(out, c_star='k7=0.01')
    with pytest.raises(AssertionError, match='names k7=0.01'):
        held(name, out)
    put_verdict(out, 'KEEP-G')
    with pytest.raises(AssertionError, match='KEEP-G'):
        held(name, out)
    assert not recorder.calls


def test_held_refuses_a_stage_3_verdict_changed_after_the_first_full_run(held_setup):
    recorder, name, out = held_setup
    put(out / VERDICT_BAD, dict(outcome='PICK', c_star=name, note='rewritten'))
    with pytest.raises(AssertionError, match='changed after'):
        held(name, out)
    assert not recorder.calls


def test_held_refuses_a_missing_seed_before_writing_anything(held_setup):
    recorder, name, out = held_setup
    rb_student.weights_path(paths.full_root(out), name, 'style', 3).unlink()
    with pytest.raises(AssertionError, match='missing'):
        held(name, out)
    assert not recorder.calls
    with pytest.raises(AssertionError, match='all seeds'):
        held(name, out, seeds=(0,))


def test_held_sees_the_knobs_when_it_checks_the_weights_switches(held_setup):
    # the weights of a k4 configuration record its patched genre switches; held must patch before it compares them
    recorder, _, out = held_setup
    cfg = S.config({'k4': 0.025})
    name = S.name(cfg)
    put_verdict(out, c_star=name)
    train(name, seeds=S.FULL_SEEDS, outputs=out)
    held(name, out)
    assert recorder.calls[-1]['arms']['tau_t'] == 0.025
    path = rb_student.weights_path(paths.full_root(out), name, 'genre', 2).with_suffix('.json')
    info = json.loads(path.read_text())
    info['config']['rg']['switches'] = V.switches('GS-0.02')
    put(path, info)
    with pytest.raises(AssertionError, match='other arm or switches|weights checksum|switches'):
        held(name, out)


def test_held_refuses_after_a_failed_training_seed(held_setup):
    recorder, name, out = held_setup
    put(rg_full.failed_path(paths.full_root(out), name, 4), dict(status='failed'))
    with pytest.raises(AssertionError, match='VC4'):
        held(name, out)


def test_held_requires_the_rule_commit(held_setup):
    recorder, name, out = held_setup
    with pytest.raises(AssertionError, match='rule commit'):
        F.held(name, S.FULL_SEEDS, out, 'cpu', ancestor=lambda: False)
    assert not recorder.calls


def test_ablation_arms_need_a_verdict_whose_final_model_they_remove_a_part_from():
    cfg = S.config({'k6': 5e-4})
    ablation = S.config({'k6': 5e-4}, remove=['buddy'])
    assert F.stage3_problem(ablation, dict(outcome='PICK', c_star=S.name(cfg))) is None
    assert F.stage3_problem(S.config(remove=['buddy']), dict(outcome='KEEP-G', c_star=None)) is None
    assert 'final model' in F.stage3_problem(S.config(remove=['buddy']), dict(outcome='PICK', c_star=S.name(cfg)))
    assert 'final model' in F.stage3_problem(ablation, dict(outcome='KEEP-G', c_star=None))
    assert F.stage3_problem(cfg, dict(outcome='PICK', c_star=S.name(cfg))) is None


def test_v4c_record_refuses_wrong_rows_hash_or_missing_output(tmp_path):
    root = tmp_path / 'held/x'
    for aspect in ASPECTS:
        put(root / f'{aspect}__s0.json', dict(rows=paths.N_HELD, list_sha256=paths.HELD_SHA256, sha256='a' * 64,
                                              vc3_attempts=1, vc3_problems=[]))
    record = F.v4c_record(tmp_path, 'x', [0])
    assert record['pass'] and sorted(record['outputs']) == sorted(f'{a}__s0' for a in ASPECTS)
    info = json.loads((root / 'genre__s0.json').read_text())
    put(root / 'genre__s0.json', dict(info, list_sha256='b' * 64))
    with pytest.raises(AssertionError, match="held list"):
        F.v4c_record(tmp_path, 'x', [0])
    put(root / 'genre__s0.json', dict(info, rows=5))
    with pytest.raises(AssertionError, match='rows'):
        F.v4c_record(tmp_path, 'x', [0])
    (root / 'genre__s0.json').unlink()
    with pytest.raises(AssertionError, match='missing'):
        F.v4c_record(tmp_path, 'x', [0])


# ------------------------------------------------------------------------------------------------------------ V4a
def fake_weights(root, name, seed, *, weights_seed=0):
    """Weights files in the layout of 12-27's stage C under `root` for arm `name`."""
    for aspect in ASPECTS:
        path = rb_student.weights_path(root, name, aspect, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        gen = torch.Generator().manual_seed(weights_seed)
        torch.save({'w': torch.randn(4, 3, generator=gen), 'b': torch.randn(3, generator=gen)}, path)
        put(path.with_suffix('.json'), dict(weights_sha256=ms_train.head.sha_file(path), best_epoch=4, best_dev_loss=0.25,
                                            config=dict(device='cuda', smoke=False)))


def fake_held(root, name, seed, *, held_seed=0):
    for aspect in ASPECTS:
        out = np.random.default_rng(held_seed).standard_normal((7, 8)).astype(np.float16)
        rb_student.write_outputs(Path(root) / 'held' / name / f'{aspect}__s{seed}.npz', dict(img=out), dict(rows=7))


def fake_reference(root, name, seed, **kwargs):
    fake_weights(root, name, seed, weights_seed=kwargs.get('weights_seed', 0))
    fake_held(root, name, seed, held_seed=kwargs.get('held_seed', 0))


def test_v4a_items_pass_for_equal_files_and_fail_on_one_bit(tmp_path):
    fake_reference(tmp_path / 'ours', 'G-v4a', 0)
    fake_reference(tmp_path / 'ref', 'GS-0.02', 0)
    items = F.v4a_items(tmp_path / 'ours', tmp_path / 'ref')
    assert all(item['pass'] for item in items.values()) and set(items) == set(ASPECTS)
    assert items['genre']['weights']['ours_sha256'] == items['genre']['weights']['reference_sha256']
    fake_held(tmp_path / 'ours', 'G-v4a', 0, held_seed=1)               # held outputs differ, weights equal
    items = F.v4a_items(tmp_path / 'ours', tmp_path / 'ref')
    assert not items['genre']['pass'] and items['genre']['weights']['pass'] and not items['genre']['held']['pass']
    fake_reference(tmp_path / 'ours', 'G-v4a', 0, weights_seed=1)       # weights differ
    items = F.v4a_items(tmp_path / 'ours', tmp_path / 'ref')
    assert not items['emotion']['weights']['pass'] and items['emotion']['held']['pass']


@pytest.fixture
def v4a(run, tmp_path, monkeypatch):
    """The cohort writes, and held inference reads, the reference's bits: what a correct trainer gives."""
    ref = tmp_path / 'ref'
    fake_reference(ref, 'GS-0.02', 0)
    monkeypatch.setattr(F, 'REFERENCE', ref)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    held_calls = []
    original = run.cohort

    def cohort(s, inputs, graphs, manifests, images, arm, seed, device, root, **kwargs):
        original(s, inputs, graphs, manifests, images, arm, seed, device, root, **kwargs)
        fake_weights(root, arm, seed)
        return {}

    def infer(arm, seeds, device, root, *, smoke=False):
        fake_held(root, arm, 0)
        held_calls.append((arm, list(seeds), Path(root)))
        return {}

    monkeypatch.setattr(rg_full, 'train_cohort', cohort)
    monkeypatch.setattr(rg_full, 'infer_held', infer)
    return held_calls


def test_regress_v4a_trains_g_into_the_regression_folder_and_writes_the_gate(run, v4a, tmp_path):
    record = F.regress_v4a('cuda', tmp_path)
    assert v4a == [('G-v4a', [0], (tmp_path / 'regression/v4a_run').resolve())]
    assert record['pass'] is True and all(item['pass'] for item in record['items'].values())
    gate = json.loads((tmp_path / 'gates/v4a_train.json').read_text())
    assert gate['pass'] is True and gate['smoke'] is False and gate['code_sha256'] == F.code_sha256()
    assert gate['frozen_sha256'] == F.require_frozen_code() and gate['frozen_commit'] == paths.FROZEN_COMMIT
    assert run.calls[0]['arms'] == V.ARMS['GS-0.02'] and run.calls[0]['lr'] == 1e-3
    assert run.calls[0]['device'] == 'cuda'


def test_regress_v4a_fails_when_one_held_bit_differs(run, v4a, tmp_path, monkeypatch):
    monkeypatch.setattr(rg_full, 'infer_held', lambda arm, seeds, device, root, *, smoke=False: fake_held(root, arm, 0, held_seed=1))
    record = F.regress_v4a('cuda', tmp_path)
    assert record['pass'] is False and not record['items']['genre']['held']['pass']
    assert json.loads((tmp_path / 'gates/v4a_train.json').read_text())['pass'] is False


def test_regress_v4a_smoke_cannot_pass(run, v4a):
    out = scratch('r10-rs-08-test-v4a')
    record = F.regress_v4a('cuda', out, smoke=True)
    assert record['pass'] is False and all(item['pass'] for item in record['items'].values())
    assert json.loads((out / 'gates/v4a_train.json').read_text())['smoke'] is True


def test_regress_v4a_runs_on_the_device_of_the_reference(run, tmp_path):
    with pytest.raises(AssertionError, match='trained on cuda'):
        F.regress_v4a('cpu', tmp_path)
    assert not run.calls
