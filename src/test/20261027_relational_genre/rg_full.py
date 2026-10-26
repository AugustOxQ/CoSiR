"""Full-scale trainer, held inference with its guard, and the training half of VC1 (rule §2.3, §4.3).

`train`    an arm at full scale: 12-24's part 2 stage 4 path (`rb_student.train` and `train_cohort`, Qwen, recipe R6-5,
           the CT's 42,969-node graphs, early stopping on the 6,152 validation paintings) with this round's genre
           switches. Only the loss differs: `rg_loss.switched_head_loss` is patched as `ms_train.head_loss` in place of
           `rb_student.loss_patch`; with every switch off it calls the AB loss exactly as `loss_patch` did. Files under
           `rg_paths.full_root(out)`: `weights/<arm>/`, `train/<arm>/`, `held/<arm>/` as 12-24's layout.
`held`     the held outputs (float16, 2048 wide, H's order) of an arm's three best views: a copy of
           `rb_student.infer_arm` for kind held, behind this round's guard (`require_guard`, in place of 12-24's gate).
`regress vc1`  R6-5 (every switch off) at full scale, seed 0, into `stage_c/regression/vc1_run/`; its weights and its
           held outputs equal 12-24's K4 files bit for bit; writes `stage_c/regression/vc1_train.json`.
"""
import copy
import hashlib
import json
import os
import subprocess
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import rg_paths
import rg_variants as V
import rg_loss
import rb_encoders
import rb_pick
import rb_rows
import rb_student
import ct2_d2
import ct2_data
import ms_train

ASPECTS = rb_student.ASPECTS
ENCODER = 'qwen3vl_emb_2b'
RECIPE = V.BASE                                         # 'R6-5': every arm of this round is R6-5 plus genre switches
SCALE = 'full'
OUT = rb_student.OUT
K4_FULL = rg_paths.RB_PART2 / 'full'                    # 12-24's frozen K4 (Qwen, R6-5) weights and held outputs
SEEDS = tuple(range(5))
CODE_FILES = ('rg_loss.py', 'rg_variants.py', 'rg_full.py')   # the files whose change alters a trained weight
VC1_RUN = Path('stage_c/regression/vc1_run')              # under the outputs folder: `regress vc1`'s weights and held outputs


class NonFiniteLoss(Exception):
    """A train or dev loss that is not finite; args[0] is the aspect (the run stops, VC4)."""


# --- the loss wiring ----------------------------------------------------------------------------------------------

def genre_for(arm, seed, *, fit_teacher, n_dev):
    """The run's `GenreLoss`. GQ arms: 4,096 anchors drawn by `rg_loss.draw_anchors` (its own generator: the global
    streams stay as they are) from the fit paintings' teacher rows `fit_teacher` (torch or numpy, float32 [n_fit, 2048],
    the genre aspect's instructed image of the fit paintings in the trainer's fit order); `n_dev`: the dev tensor's
    row count. The loss re-checks on every fit call that the anchors equal the teacher tensor's rows."""
    switches = V.switches(arm)
    if not switches['anchors']:
        return rg_loss.GenreLoss(switches, seed)
    assert fit_teacher is not None and n_dev is not None, f'{arm}: anchors need the fit teacher rows and n_dev'
    fit_teacher = torch.as_tensor(fit_teacher)
    assert fit_teacher.ndim == 2 and fit_teacher.dtype == torch.float32 and fit_teacher.device.type == 'cpu'
    ids = rg_loss.draw_anchors(len(fit_teacher), seed)
    return rg_loss.GenreLoss(switches, seed, anchors=fit_teacher[ids].clone(), anchor_ids=ids,
                             n_fit=len(fit_teacher), n_dev=n_dev)


def checked_head_loss(*args, genre=None, **kwargs):
    """`rg_loss.switched_head_loss`, stopping the run on a loss that is not finite (train and dev calls alike)."""
    loss = rg_loss.switched_head_loss(*args, genre=genre, **kwargs)
    if not bool(torch.isfinite(loss)):
        raise NonFiniteLoss(kwargs['aspect'] if 'aspect' in kwargs else args[6])
    return loss


def head_loss_for(arm, seed, *, fit_teacher, n_dev):
    """What `ms_train.head_loss` is patched to for the arm (in place of `rb_student.loss_patch`). Switches off: the AB
    loss with R6-5's switches, as `loss_patch` patches it. Genre switches on: the genre head's loss changes only."""
    genre = genre_for(arm, seed, fit_teacher=fit_teacher, n_dev=n_dev)
    return partial(checked_head_loss, switches=rb_rows.recipe(RECIPE)['switches'], genre=genre)


# --- the held guard -----------------------------------------------------------------------------------------------

def rule_commit_is_ancestor(commit=rg_paths.RULE_COMMIT):
    """True when the rule's commit is an ancestor of this checkout's HEAD (an unknown commit: False)."""
    done = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=rg_paths.ROUND,
                          capture_output=True)
    return done.returncode == 0


def _record(path):
    path = Path(path)
    assert path.is_file(), f'{path}: missing'
    record = json.loads(path.read_text())
    assert isinstance(record, dict), f'{path}: not a JSON object'
    return record


def _weights_record(root, arm, aspect, seed):
    return _record(rb_student.weights_path(root, arm, aspect, seed).with_suffix('.json'))


def require_guard(out, arm, *, ancestor=rule_commit_is_ancestor, seeds=V.FULL_SEEDS):
    """The held guard (rule §4.3 VC1, VC4): held outputs of `arm` only when `stage_c/regression/vc1_train.json` (a full
    run, not a smoke) and `vc1_scorer.json` pass, the weights of `arm` for `seeds` were trained by the code VC1 checked
    (the `rg` and `rb` code hashes of their records equal those of the VC1 run's R6-5 weights), the code on disk now
    has those hashes too, the VC1 run's weights are the files `vc1_train.json` checked, `stage_b/verdict.json`
    is B-PICK with `v_held` == `arm` and its switches equal the arm's, and the rule commit is an ancestor of HEAD.
    Returns the SHA-256 of the three files read."""
    out = Path(out)
    assert arm in V.ARMS and arm != V.BASE, f'{arm}: not a candidate arm of this round'
    shas = {}
    for name in ('vc1_train', 'vc1_scorer'):
        path = out / f'stage_c/regression/{name}.json'
        assert path.is_file(), f'VC1: {path} missing'
        record = _record(path)
        assert record.get('pass') is True, f'VC1: {path} does not pass'
        if name == 'vc1_train':
            assert record.get('identity', {}).get('smoke') is False, f'VC1: {path} is a smoke run (or has no identity)'
        shas[name] = rb_student.ms_train.head.sha_file(path)
    path = out / 'stage_b/verdict.json'
    verdict = _record(path)
    assert verdict.get('outcome') == 'B-PICK', f'{path}: outcome {verdict.get("outcome")!r}; held needs B-PICK'
    assert verdict.get('v_held') == arm, f'{path}: v_held is {verdict.get("v_held")!r}; G is {arm!r}'
    assert verdict.get('v_held_switches') == V.switches(arm), f'{path}: v_held_switches differ from {arm}\'s switches'
    shas['stage_b_verdict'] = rb_student.ms_train.head.sha_file(path)
    now = dict(rg=_code_sha256(), rb=_rb_code_sha256())                  # the code on disk now, not the records' claim
    items = _record(out / 'stage_c/regression/vc1_train.json').get('items', {})
    for aspect in ASPECTS:
        vc1_record = _weights_record(out / VC1_RUN, V.BASE, aspect, 0)
        vc1 = vc1_record['config']
        ours = items.get(aspect, {}).get('weights', {}).get('ours_sha256')
        assert ours is not None and vc1_record.get('weights_sha256') == ours, \
            f'VC1: the VC1 run\'s {aspect} weights are not the ones vc1_train.json checked; rerun `regress vc1`'
        for block in ('rg', 'rb'):
            assert vc1[block].get('code_sha256') == now[block], \
                f'VC1: the {block} code on disk now is not the {block} code of the VC1 run; rerun `regress vc1` and retrain {arm}'
        for seed in seeds:
            config = _weights_record(rg_paths.full_root(out), arm, aspect, seed)['config']
            for block in ('rg', 'rb'):
                assert config[block].get('code_sha256') == vc1[block]['code_sha256'], \
                    f'VC1: {arm} {aspect} s{seed} was trained by other {block} code than the VC1 run checked; rerun `regress vc1`'
    assert ancestor(), f'the rule commit {rg_paths.RULE_COMMIT} is not an ancestor of HEAD'
    return shas


# --- provenance ---------------------------------------------------------------------------------------------------

def code_provenance():
    """git HEAD of this checkout, whether this round's .py files are modified or new, and the hashes of this round's code."""
    git = ['git', '-C', str(rg_paths.ROUND)]
    head = subprocess.run(git + ['rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    # this round's .py files, untracked ones included (a new module not yet committed counts as dirty), as rg_held
    dirty = bool(subprocess.run(git + ['status', '--porcelain', '--', '*.py'], cwd=rg_paths.ROUND,
                                capture_output=True, text=True, check=True).stdout.strip())
    return dict(code_commit=head, dirty=dirty, rg_full_sha256=ms_train.head.sha_file(Path(__file__)),
                rg_loss_sha256=ms_train.head.sha_file(rg_paths.ROUND / 'rg_loss.py'))


def _code_sha256():
    sha = {name: ms_train.head.sha_file(rg_paths.ROUND / name) for name in CODE_FILES}
    return hashlib.sha256(json.dumps(sha, sort_keys=True).encode()).hexdigest()


def _rb_code_sha256():
    """The `rb` block's `code_sha256` for the code on disk now: `rb_student._rb_block`'s formula over `code_hashes()`."""
    return hashlib.sha256(json.dumps(rb_student.code_hashes(), sort_keys=True).encode()).hexdigest()


def rg_block(arm, seed, n_fit):
    """The config's rg part: the arm's switches and, for GQ, the anchors' positions (count, SHA-256)."""
    switches = V.switches(arm)
    anchors = None
    if switches['anchors']:
        ids = rg_loss.draw_anchors(n_fit, seed)
        anchors = dict(count=int(len(ids)), n_fit=int(n_fit), positions_sha256=rb_student.sha_array(ids.numpy()))
    return dict(arm=arm, switches=switches, anchors=anchors, code_sha256=_code_sha256())


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _check_root(root):
    """Nothing of this round writes into a frozen folder (12-24, 12-25, the CT)."""
    root = Path(root).resolve()
    for frozen in (rg_paths.EG_OUT, rg_paths.RB_OUT, rg_paths.CT_OUT):
        assert not root.is_relative_to(Path(frozen).resolve()), f'{root} is inside the frozen {frozen}'
    return root


# --- training -----------------------------------------------------------------------------------------------------

def failed_path(root, arm, seed):
    return Path(root) / 'train' / arm / f'failed__s{seed}.json'


def _with_blocks(make_config, rb, rg):
    """`rb_student._with_rb` plus this round's rg block (the arm's switches) in every config."""
    def wrapped(*args, **kwargs):
        config = make_config(*args, **kwargs)
        assert 'rb' not in config and 'rg' not in config
        config['rb'], config['rg'] = copy.deepcopy(rb), copy.deepcopy(rg)
        return config
    return wrapped


def train_cohort(s, inputs, graphs, manifests, images, arm, seed, device, root, *, rb, smoke=False, max_epochs=None):
    """One three-aspect cohort of `arm` at full scale: `rb_student.train_cohort` for Qwen and R6-5, with the arm's loss.
    Returns the loop's result, or a failed record (also written to `failed_path`) when a loss is not finite."""
    recipe = rb_rows.recipe(RECIPE)
    assert recipe['base']['inputs'] == 'default', 'the student takes the default inputs (Qwen caption mean)'
    width = rb_encoders.WIDTH[ENCODER]
    assert images.shape == (len(s['first']), width) and images.dtype == np.float32 and np.isfinite(images).all()
    assert all(inputs[a]['cap'].shape == (len(s['first']), 2048) for a in ASPECTS)
    inputs = {a: dict(img=images, cap=inputs[a]['cap']) for a in ASPECTS}
    max_epochs = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    fit_teacher = None
    if V.switches(arm)['anchors']:
        # The genre teacher as ct2_d2.run builds it (default inputs: the instructed genre image of every painting),
        # restricted to the fit paintings in the trainer's fit order; run's tensors are the same rows.
        teacher = ct2_data.instructed_images(s, ms_train.head.ASPECTS['genre'])
        assert teacher.shape == (len(s['first']), 2048) and teacher.dtype == np.float32
        assert rb_student.sha_array(teacher) == s['data_sha256']['genre.img'], 'genre teacher differs from the CT\'s'
        fit_teacher = torch.from_numpy(np.ascontiguousarray(teacher[s['fit']]))
        del teacher
    rg = rg_block(arm, seed, len(s['fit']))
    loss = head_loss_for(arm, seed, fit_teacher=fit_teacher, n_dev=len(s['dev']))
    start = time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(patch.object(ms_train, 'build_model', partial(rb_student.build_model, width=width)))
        stack.enter_context(patch.object(ms_train, 'head_loss', loss))
        stack.enter_context(patch.object(ct2_d2, 'make_config', _with_blocks(rb_student.CT2_MAKE_CONFIG, rb, rg)))
        stack.enter_context(patch.object(ct2_d2, 'save_output',
                                         rb_student._weights_saver(rb_student.CT2_SAVE_OUTPUT, root, arm)))
        try:
            return ct2_d2.run(s, inputs, graphs, manifests, recipe['base'], RECIPE, list(ASPECTS), seed, device,
                              Path(root) / 'train' / arm, smoke=smoke, max_epochs=max_epochs)
        except NonFiniteLoss as error:
            record = dict(arm=arm, seed=seed, status='failed', failure_kind='nonfinite_loss',
                          cause=f'non-finite loss in the {error.args[0]} head', switches=V.switches(arm),
                          smoke=smoke, max_epochs=max_epochs, seconds=time.monotonic() - start,
                          **code_provenance())
            _atomic_json(failed_path(root, arm, seed), record)
            print(f'{arm} s{seed}: {record["cause"]} (VC4: no rerun)', flush=True)
            return record


def _cohort_state(root, arm, seed, rb, rg):
    """None (nothing there), 'done' (verified weights of this arm's rb and rg blocks) or 'broken'."""
    state = rb_student._incomplete_cohort(root, arm, seed, RECIPE, SCALE, rb)
    if state == 'done':
        for aspect in ASPECTS:
            info = json.loads(rb_student.weights_path(root, arm, aspect, seed).with_suffix('.json').read_text())
            assert info['config']['rg'] == rg, f'{arm} s{seed} {aspect}: weights of other switches or code; move them aside'
    return state


@threadpool_limits.wrap(limits=8)
def train(arm, seeds, device, outputs, *, smoke=False, max_epochs=None, root=None):
    """Train `arm` for each seed at full scale into `root` (default `rg_paths.full_root(outputs)`). A seed with
    verified weights is skipped; an arm with any held output is not trained (VC4); a seed that stopped on a non-finite
    loss is not rerun (VC4)."""
    seeds = list(seeds)
    assert arm in V.ARMS and seeds and len(set(seeds)) == len(seeds) and all(s in SEEDS for s in seeds)
    assert arm != V.BASE or root is not None, f'{V.BASE} trains only as the VC1 regression (`regress vc1`)'
    outputs, device = rb_student._check_outputs(outputs, smoke), torch.device(device)
    assert not smoke or (seeds == [0] and (max_epochs or 2) <= 2), '--smoke: seed 0 and at most 2 epochs'
    if device.type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (CT queue)'
    root = _check_root(root if root is not None else rg_paths.full_root(outputs))
    for seed in seeds:     # refuse before the data load
        assert not failed_path(root, arm, seed).exists(), \
            f'VC4: {arm} s{seed} stopped on a non-finite loss ({failed_path(root, arm, seed)}); no rerun'
    torch.set_num_threads(8)
    s, inputs, graphs, manifests = rb_student.load_scale(SCALE)
    images = rb_student.training_images(ENCODER, s, inputs)
    rb = rb_student._rb_block(SCALE, ENCODER, RECIPE, rb_rows.recipe(RECIPE), images)
    result = {}
    for seed in seeds:
        rg = rg_block(arm, seed, len(s['fit']))
        state = _cohort_state(root, arm, seed, rb, rg)
        if state == 'done':
            print(f'{arm} s{seed}: verified weights; skip', flush=True)
            continue
        held = sorted((root / 'held' / arm).rglob('*.npz')) if (root / 'held' / arm).exists() else []
        assert not held, f'VC4: no new training of {arm} once held outputs exist (any seed): {held[0]}'
        if state == 'broken':
            rb_student._move_aside(root, arm, seed, RECIPE)
        result[seed] = train_cohort(s, inputs, graphs, manifests, images, arm, seed, device, root, rb=rb,
                                    smoke=smoke, max_epochs=max_epochs)
    return result


# --- held inference -----------------------------------------------------------------------------------------------

def vc3_problem(x, n_rows=rg_paths.N_HELD):
    """VC3 on one output set: None, or the first problem (rows, dtype, finite, unit norm); 12-24's VP2 check."""
    return rb_pick.vp2_problem(x, n_rows)


def _arm_weights(root, arm, aspect, seed, smoke):
    """(path, config, weights sha) of an arm's saved best view, checked against its record and its switches."""
    path = rb_student.weights_path(root, arm, aspect, seed)
    assert path.is_file(), f'{path}: missing (train {arm} first)'
    info = json.loads(path.with_suffix('.json').read_text())
    assert ms_train.head.sha_file(path) == info['weights_sha256'], f'{path}: weights checksum mismatch'
    config = info['config']
    assert (config['aspect'], config['seed'], config['smoke']) == (aspect, seed, smoke)
    assert smoke or config['max_epochs'] == ms_train.CONFIG['max_epochs'], \
        f'{path}: trained with max_epochs {config["max_epochs"]}, not the CT\'s {ms_train.CONFIG["max_epochs"]}'
    assert (config['rb']['scale'], config['rb']['encoder'], config['rb']['width'], config['rb']['recipe']) == (
        SCALE, ENCODER, rb_encoders.WIDTH[ENCODER], RECIPE)
    assert config['rg']['arm'] == arm and config['rg']['switches'] == V.switches(arm), f'{path}: other arm or switches'
    return path, config, info['weights_sha256']


def _refuse_failed(root, arm, seeds):
    """VC4: a seed whose training stopped on a non-finite loss, or whose held output failed VC3 twice, is not tried again."""
    for seed in seeds:
        assert not failed_path(root, arm, seed).exists(), \
            f'VC4: {arm} s{seed} stopped on a non-finite loss ({failed_path(root, arm, seed)}); no held output'
        for aspect in ASPECTS:
            path = root / 'held' / arm / f'failed__{aspect}__s{seed}.json'
            assert not path.exists(), f'VC4: {arm} {aspect} s{seed} failed VC3 twice ({path}); no new attempt'


def infer_held(arm, seeds, device, root, *, smoke=False):
    """A copy of `rb_student.infer_arm` for kind held, Qwen at full scale. No guard here (`held` and `regress vc1`
    decide when this may run). Files `root/held/<arm>/<aspect>__s{seed}.npz` as 12-24's; an existing file must match.
    VC3 on each output; a failure is rerun once from the same weights, and a second failure is recorded in
    `root/held/<arm>/failed__<aspect>__s{seed}.json` and stops; the output then keeps the first attempt's problem in
    `vc3_problems`. A seed with such a record, or with a training run that stopped on a non-finite loss, is refused
    (VC4: no stage C outcome, no new attempt), before any file is written."""
    seeds = list(seeds)
    assert arm in V.ARMS and seeds and len(set(seeds)) == len(seeds) and all(s in SEEDS for s in seeds)
    root = _check_root(root)
    _refuse_failed(root, arm, seeds)
    names, names_sha = rb_student.held_names()
    x = rb_encoders.full_images(ENCODER, names)
    width = rb_encoders.WIDTH[ENCODER]
    assert x.shape == (len(names), width) == (rg_paths.N_HELD, width) and x.dtype == np.float32
    assert np.isfinite(x).all()
    result = {}
    for seed in seeds:
        for aspect in ASPECTS:
            weights, config, weights_sha = _arm_weights(root, arm, aspect, seed, smoke)
            recorded = torch.device(config['device'])
            assert device is None or torch.device(device) == recorded, 'inference device differs from training device'
            path = root / 'held' / arm / f'{aspect}__s{seed}.npz'
            meta = dict(kind='held', arm=arm, scale=SCALE, aspect=aspect, seed=seed, encoder=ENCODER, rows=len(names),
                        width=OUT, dtype='float16', list_sha256=names_sha, weights_sha256=weights_sha,
                        image_sha256=rb_student.sha_array(x), device=str(recorded), recipe=RECIPE,
                        switches=V.switches(arm))
            if path.exists():
                arrays, info = rb_student.read_outputs(path)
                assert all(info[k] == v for k, v in meta.items()), f'{path}: existing output differs from this run'
                out = arrays['img']
                problem = vc3_problem(out)
                assert problem is None, f'{path}: VC3: {problem}'
            else:
                problems = []
                for attempt in (1, 2):
                    try:
                        out = rb_student.infer(weights, aspect, width, x, recorded)
                        problem = vc3_problem(out)
                    except AssertionError as error:       # the inputs are checked above: this is the output's check
                        problem = f'infer: {error}'
                    if problem is None:
                        break
                    problems.append(problem)
                    print(f'{path}: VC3 failure ({problem}); {"rerun once" if attempt == 1 else "stop"}', flush=True)
                else:
                    _atomic_json(path.parent / f'failed__{aspect}__s{seed}.json',
                                 dict(arm=arm, aspect=aspect, seed=seed, failure_kind='vc3', problems=problems,
                                      **code_provenance()))
                    raise AssertionError(f'{path}: VC3 failed twice: {problems}')
                rb_student.write_outputs(path, dict(img=out),
                                         dict(meta, sha256=rb_student.sha_array(out), vc3_attempts=len(problems) + 1,
                                          vc3_problems=problems))
                arrays, info = rb_student.read_outputs(path)
            assert info['sha256'] == info['arrays']['img']['sha256'] == rb_student.sha_array(out)
            result[aspect, seed] = out
    return result


@threadpool_limits.wrap(limits=8)
def held(arm, seeds, outputs, device=None, *, smoke=False, ancestor=rule_commit_is_ancestor):
    """Guarded held inference of `arm` (G): `require_guard` first, before any data or weights are read."""
    outputs = rb_student._check_outputs(outputs, smoke)
    if not smoke:      # a smoke writes under /tmp/claude-0 only, as rb_student.infer_arm
        seeds = list(seeds)
        assert sorted(seeds) == list(V.FULL_SEEDS), f'held runs all seeds {V.FULL_SEEDS} of the arm at once, not {seeds}'
        require_guard(outputs, arm, ancestor=ancestor, seeds=seeds)
        # nothing is written before every seed's weights are there and intact (a partial run would leave held outputs
        # that block training the rest, VC4)
        root = _check_root(rg_paths.full_root(outputs))
        _refuse_failed(root, arm, seeds)
        for seed in seeds:
            for aspect in ASPECTS:
                _arm_weights(root, arm, aspect, seed, False)
    else:
        assert arm in V.ARMS and arm != V.BASE, arm
    torch.set_num_threads(8)
    return infer_held(arm, seeds, device, rg_paths.full_root(outputs), smoke=smoke)


# --- VC1, the training half ---------------------------------------------------------------------------------------

def vc1_items(reg_root, k4_root=K4_FULL, seed=0):
    """Compare R6-5 trained by this module (`reg_root`) with 12-24's K4 files, bit for bit, per aspect: the weights
    (same keys in the same order, every tensor `torch.equal`), the best epoch and dev loss, and the held `img`."""
    items = {}
    for aspect in ASPECTS:
        ours_pt = rb_student.weights_path(reg_root, V.BASE, aspect, seed)
        theirs_pt = rb_student.weights_path(k4_root, 'K4', aspect, seed)
        a, b = (torch.load(p, map_location='cpu', weights_only=True) for p in (ours_pt, theirs_pt))
        same = list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
        oi, ti = (json.loads(p.with_suffix('.json').read_text()) for p in (ours_pt, theirs_pt))
        recipe_ok = (ti['config']['rb']['recipe'], ti['config']['rb']['encoder']) == (RECIPE, ENCODER)
        parts = {'weights': {'pass': bool(same), 'keys': list(a) == list(b), 'ours_sha256': oi['weights_sha256'],
                             'k4_sha256': ti['weights_sha256']},
                 'k4_recipe': {'pass': bool(recipe_ok), 'recipe': ti['config']['rb']['recipe']},
                 'best_epoch': {'pass': bool(oi['best_epoch'] == ti['best_epoch']
                                             and oi['best_dev_loss'] == ti['best_dev_loss']),
                                'ours': [oi['best_epoch'], oi['best_dev_loss']],
                                'k4': [ti['best_epoch'], ti['best_dev_loss']]}}
        ours, _ = rb_student.read_outputs(reg_root / 'held' / V.BASE / f'{aspect}__s{seed}.npz')
        theirs, _ = rb_student.read_outputs(k4_root / 'held' / 'K4' / f'{aspect}__s{seed}.npz')
        ours, theirs = ours['img'], theirs['img']
        equal = (ours.shape == theirs.shape and ours.dtype == theirs.dtype
                 and np.array_equal(rb_student._bits(ours), rb_student._bits(theirs)))
        parts['held'] = {'pass': bool(equal), 'rows': int(len(ours)), 'ours_sha256': rb_student.sha_array(ours),
                         'k4_sha256': rb_student.sha_array(theirs)}
        items[f'{aspect}'] = dict(**{'pass': all(v['pass'] for v in parts.values())}, **parts)
    return items


def regress_vc1(device, outputs, *, smoke=False):
    """R6-5 (every switch off) at full scale, seed 0 -> weights and held outputs under `stage_c/regression/vc1_run/`,
    compared with K4's; writes `stage_c/regression/vc1_train.json` ({pass, items, identity}). With --smoke the run is
    2 epochs and cannot equal K4: the record is written (pass false) only to prove the path."""
    outputs = rb_student._check_outputs(outputs, smoke)
    recorded = json.loads(rb_student.weights_path(K4_FULL, 'K4', ASPECTS[0], 0).with_suffix('.json').read_text())
    assert str(torch.device(device)) == recorded['config']['device'], \
        f'12-24 trained K4 on {recorded["config"]["device"]}; the regression must run on it'
    reg_root = outputs / 'stage_c/regression/vc1_run'
    train(V.BASE, [0], device, outputs, smoke=smoke, root=reg_root)
    infer_held(V.BASE, [0], device, reg_root, smoke=smoke)
    items = vc1_items(reg_root)
    record = dict(items=items, identity=dict(device=str(device), smoke=smoke, k4_root=str(K4_FULL), **code_provenance()))
    record['pass'] = all(item['pass'] for item in items.values()) and not smoke
    _atomic_json(outputs / 'stage_c/regression/vc1_train.json', record)
    print(f'vc1 (training half): pass={record["pass"]}; failing: {[k for k, v in items.items() if not v["pass"]]}',
          flush=True)
    return record


# --- command line -------------------------------------------------------------------------------------------------

def main(argv=None):
    import argparse
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # the CT queue's setting, before any CUDA work
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--outputs', type=Path, default=rg_paths.OUTPUTS)
        p.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0; seed 0; at most 2 epochs')

    candidates = [arm for arm in V.ARMS if arm != V.BASE]
    p = commands.add_parser('train', help='train an arm at full scale')
    p.add_argument('--arm', required=True, choices=candidates)
    p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-epochs', type=int)
    common(p)
    p = commands.add_parser('held', help='held outputs of a trained arm (guarded)')
    p.add_argument('--arm', required=True, choices=candidates)
    p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    p.add_argument('--device', help='default: the device the weights were trained on')
    common(p)
    p = commands.add_parser('regress', help='VC1, training half: R6-5 at full scale against 12-24\'s K4')
    p.add_argument('stage', choices=['vc1'])
    p.add_argument('--device', default='cuda')
    common(p)
    args = parser.parse_args(argv)
    if args.command == 'train':
        seeds = [0] if args.smoke else args.seeds
        return train(args.arm, seeds, args.device, args.outputs, smoke=args.smoke, max_epochs=args.max_epochs)
    if args.command == 'held':
        seeds = [0] if args.smoke else args.seeds
        return held(args.arm, seeds, args.outputs, args.device, smoke=args.smoke)
    return regress_vc1(args.device, args.outputs, smoke=args.smoke)


if __name__ == '__main__':
    main()
