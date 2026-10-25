"""Width-opened D2 student at the sample or full scale; held and ArtELingo-28 inference; stage 2 and 3a regressions.

A student is the CT's D2 cohort (three aspects, one optimizer) with the picked recipe's switches (own, endpoints, w)
and one change: the image network's first Linear takes the encoder width. The training loops are the CT's own
(`ms_train.run` at the sample scale, `ct2_d2.run` at the full scale); this module patches only the model factory, the
loss (the AB's switched loss, when a switch is off), the config (adds an `rb` block) and the saver (best weights).
Qwen with R0-5 therefore runs the CT's computation unchanged (the regressions check it bit for bit).

Files below OUTPUTS/part2 (rb index): `<scale>/weights/<K>/<aspect>__s{s}.pt|json`,
`<scale>/held/<K>/<aspect>__s{s}.npz|json`, `<scale>/a28/<K>/...`, `regression/{stage2,stage3a}.json`.
"""
import copy
import hashlib
import json
import os
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import rb_paths
import rb_rows
import rb_encoders
import rb_score
import ab_train
import ct_d2
import ct2_d2
import ct_held
import ms_train

ASPECTS = ct_d2.ASPECTS
ARM = {'siglip2_so400m': 'K2', 'clip_b32': 'K3', 'qwen3vl_emb_2b': 'K4', 'dinov2_large': 'K5'}
ENCODER = {arm: encoder for encoder, arm in ARM.items()}
SCALES = ('sample', 'full')
SHEET_BUILD_MODEL = ms_train.build_model                 # captured before any patch
SHEET_HEAD_LOSS = ms_train.head_loss
HIDDEN, OUT = ms_train.CONFIG['architecture'][1:]
assert (HIDDEN, OUT, ms_train.CONFIG['architecture'][0]) == (1024, 2048, 2048)
SMOKE_ROOT = Path('/tmp/claude-0')
REGRESSION_ARM = 'regression'
STAGES = {'stage2': 'sample', 'stage3a': 'full'}
CODE_FILES = ('rb_student.py', 'rb_rows.py', 'rb_encoders.py')


class WideTrunkSide(nn.Module):
    """ms_train.TrunkSide with the first Linear taking `width`; layers are created in the same order."""

    def __init__(self, aspects, width):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(width, HIDDEN), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({a: nn.Linear(HIDDEN, OUT) for a in aspects})


def build_model(row, aspects, device, *, width):
    """The sheet's factory with the image trunk opened to `width`; at 2048 it is the sheet's factory itself."""
    assert row['share'] == 'trunk' and row['form'] == 'a', 'only the trunk-shared form-a cohort is a student'
    if width == ms_train.CONFIG['architecture'][0]:
        return SHEET_BUILD_MODEL(row, aspects, device)
    models = nn.ModuleDict({'img': WideTrunkSide(aspects, width).to(device),
                            'cap': ms_train.TrunkSide(aspects).to(device)})
    views = {a: nn.ModuleDict({side: ms_train.TrunkHead(models[side], a) for side in ('img', 'cap')})
             for a in aspects}
    return models, views


def loss_patch(switches):
    """The AB's switched loss in place of the sheet's, only when a switch is off (as `ab_train.run_cohort`)."""
    assert switches['teacher'] == 'instructed' and switches['buddy'] == 1.0, 'part 1 recipes switch own and endpoints only'
    stack = ExitStack()
    if switches != ab_train.DEFAULT_SWITCHES:
        stack.enter_context(patch.object(ms_train, 'head_loss', partial(ab_train.switched_head_loss, switches=switches)))
    return stack


# --- training ---------------------------------------------------------------------------------------------------

SHEET_MAKE_CONFIG = ms_train.make_config
SHEET_SAVE_OUTPUT = ms_train.save_output
CT2_MAKE_CONFIG = ct2_d2.make_config
CT2_SAVE_OUTPUT = ct2_d2.save_output


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def code_hashes():
    """SHA-256 of each module that writes a student file (this one, the rows, the encoders, the AB loss and rows)."""
    files = [rb_paths.ROUND / name for name in CODE_FILES] + [Path(ab_train.__file__), Path(rb_rows.ab_rows.__file__)]
    return {path.name: ms_train.head.sha_file(path) for path in files}


def arm_for(encoder, recipe_name):
    """The arm K2 to K5 of an encoder's student; K4 exists only for a recipe other than R0-5, K5 only image-only."""
    arm = ARM[encoder]
    assert arm != 'K4' or recipe_name != rb_rows.REFERENCE, 'K4 exists only when the pick is not P-KEEP (R0-5)'
    assert arm != 'K5' or recipe_name in rb_rows.IMAGE_ONLY, 'K5 exists only under an image-only recipe'
    return arm


def weights_path(root, arm, aspect, seed):
    assert aspect in ASPECTS and seed in range(5)
    return Path(root) / 'weights' / arm / f'{aspect}__s{seed}.pt'


def _rb_block(scale, encoder, recipe_name, recipe, images):
    return dict(scale=scale, encoder=encoder, width=rb_encoders.WIDTH[encoder], recipe=recipe_name,
                switches=copy.deepcopy(recipe['switches']), image_sha256=sha_array(images),
                code_sha256=hashlib.sha256(json.dumps(code_hashes(), sort_keys=True).encode()).hexdigest())


def _with_rb(make_config, rb):
    def wrapped(*args, **kwargs):
        config = make_config(*args, **kwargs)
        assert 'rb' not in config
        config['rb'] = copy.deepcopy(rb)
        return config
    return wrapped


def _weights_saver(original, root, arm):
    """The CT's weight saver (`ct_d2.train.save_weights`) around a sheet or stage 2 `save_output`."""
    def save_weights(s, view, inputs, state, path, config, start):
        info = original(s, view, inputs, state, path, config, start)
        weight = weights_path(root, arm, config['aspect'], config['seed'])
        weight.parent.mkdir(parents=True, exist_ok=True)
        pending = weight.with_suffix('.pending.pt')
        torch.save({key: value.detach().cpu() for key, value in state['best_state'].items()}, pending)
        pending.replace(weight)
        ms_train.head.atomic_json(weight.with_suffix('.json'), dict(
            weights_sha256=ms_train.head.sha_file(weight), output_sha256=info['output_sha256'],
            config=config, best_epoch=info['best_epoch'], best_dev_loss=info['best_dev_loss']))
        return info
    return save_weights


def train_cohort(scale, s, inputs, graphs, manifests, images, encoder, recipe_name, seed, device, root, arm,
                 smoke=False, max_epochs=None):
    """One three-aspect cohort of a student on `images` (the encoder image of every painting, `s` order).

    The CT's loop for the scale: sample `ms_train.run` (stage 1), full `ct2_d2.run` (stage 2). `inputs` are the CT's
    (Qwen image, default caption mean); only the image is replaced. Writes `root/train/<arm>/<recipe>/` (the loop's
    own output on the old 5,000 or the validation paintings) and `root/weights/<arm>/` (best states).
    """
    assert scale in SCALES and encoder in rb_encoders.WIDTH
    recipe = rb_rows.recipe(recipe_name)
    assert recipe['base']['inputs'] == 'default', 'the student takes the default inputs (Qwen caption mean)'
    width = rb_encoders.WIDTH[encoder]
    assert images.shape == (len(s['first']), width) and images.dtype == np.float32 and np.isfinite(images).all()
    assert all(inputs[a]['cap'].shape == (len(s['first']), 2048) for a in ASPECTS)
    inputs = {a: dict(img=images, cap=inputs[a]['cap']) for a in ASPECTS}
    max_epochs = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    rb = _rb_block(scale, encoder, recipe_name, recipe, images)
    with ExitStack() as stack:
        stack.enter_context(patch.object(ms_train, 'build_model', partial(build_model, width=width)))
        stack.enter_context(loss_patch(recipe['switches']))
        if scale == 'sample':
            run = ms_train.run
            stack.enter_context(patch.object(ms_train, 'make_config', _with_rb(SHEET_MAKE_CONFIG, rb)))
            stack.enter_context(patch.object(ms_train, 'save_output', _weights_saver(SHEET_SAVE_OUTPUT, root, arm)))
        else:
            run = ct2_d2.run
            stack.enter_context(patch.object(ct2_d2, 'make_config', _with_rb(CT2_MAKE_CONFIG, rb)))
            stack.enter_context(patch.object(ct2_d2, 'save_output', _weights_saver(CT2_SAVE_OUTPUT, root, arm)))
        return run(s, inputs, graphs, manifests, recipe['base'], recipe_name, list(ASPECTS), seed, device,
                   Path(root) / 'train' / arm, smoke=smoke, max_epochs=max_epochs)


def load_scale(scale):
    """The CT's training data of the scale: sample (stage 1 setup) or full (stage 2, graphs under `CT_FULL/d2/graphs`).

    Returns s, inputs {aspect: {img, cap}}, graphs, manifests. The full loader only reads: the graph manifest exists.
    """
    assert scale in SCALES
    if scale == 'sample':
        return ct_d2.load_training_data()
    assert (rb_paths.CT_FULL / 'd2/graphs/manifest.json').is_file()
    return ct2_d2.load_training_data('full', rb_paths.CT_FULL)


def training_images(encoder, s, inputs):
    """The encoder's default image of every painting of `s`, by name; Qwen must equal the CT's own array."""
    x = rb_encoders.full_images(encoder, s['all_painting_ids'])
    if encoder == 'qwen3vl_emb_2b':
        assert all(np.array_equal(x, inputs[a]['img']) for a in ASPECTS), 'Qwen image differs from the CT loader'
    return x


def _check_outputs(outputs, smoke):
    outputs = Path(outputs).resolve()
    if smoke:
        assert outputs.is_relative_to(SMOKE_ROOT), '--smoke outputs must be under /tmp/claude-0'
    return outputs


def _incomplete_cohort(root, arm, seed, recipe_name, scale, rb):
    """None when nothing exists; 'done' when all three aspects hold identical verified weights."""
    present = {a: weights_path(root, arm, a, seed).exists() for a in ASPECTS}
    if not any(present.values()):
        return None
    if all(present.values()):
        for aspect in ASPECTS:
            path = weights_path(root, arm, aspect, seed)
            info = json.loads(path.with_suffix('.json').read_text())
            if info['weights_sha256'] != ms_train.head.sha_file(path):
                return 'broken'
            assert info['config']['rb'] == rb, f'{path}: weights of another recipe, encoder or code; move them aside'
        return 'done'
    return 'broken'


def _move_aside(root, arm, seed, recipe_name):
    """Keep incomplete artifacts out of the loop's resume checks (the CT's `_preserve_cohort`)."""
    backup = Path(root) / 'replaced' / arm / f's{seed}-{time.time_ns()}'
    paths = []
    for aspect in ASPECTS:
        weight = weights_path(root, arm, aspect, seed)
        paths += [weight, weight.with_suffix('.json')]
    train_dir = Path(root) / 'train' / arm / recipe_name
    paths += sorted(train_dir.glob(f'*__s{seed}.*')) if train_dir.exists() else []
    for path in paths:
        if path.exists():
            target = backup / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)


@threadpool_limits.wrap(limits=8)
def train(scale, encoder, recipe_name, seeds, device, outputs, *, smoke=False, max_epochs=None, arm=None, root=None):
    """Train the student of `encoder` with recipe `recipe_name` for each seed into `root` (default part2/<scale>)."""
    seeds = list(seeds)
    assert scale in SCALES and encoder in rb_encoders.WIDTH and recipe_name in rb_rows.RECIPES
    assert seeds and len(set(seeds)) == len(seeds) and all(s in range(5) for s in seeds)
    outputs, device = _check_outputs(outputs, smoke), torch.device(device)
    assert not smoke or (seeds == [0] and (max_epochs or 2) <= 2), '--smoke: seed 0 and at most 2 epochs'
    if device.type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (CT queue)'
    arm = arm or arm_for(encoder, recipe_name)
    pick = outputs / 'part1/pick.json'
    if arm in ('K4', 'K5') and pick.exists():     # rule §4.1: the pick decides the recipe of K4 and K5
        assert recipe_name == picked_recipe(outputs), \
            f'{arm} trains the recipe of {pick} ({picked_recipe(outputs)}), not {recipe_name}'
    root = Path(root) if root is not None else outputs / 'part2' / scale
    torch.set_num_threads(8)
    s, inputs, graphs, manifests = load_scale(scale)
    images = training_images(encoder, s, inputs)
    rb = _rb_block(scale, encoder, recipe_name, rb_rows.recipe(recipe_name), images)
    result = {}
    for seed in seeds:
        state = _incomplete_cohort(root, arm, seed, recipe_name, scale, rb)
        if state == 'done':
            print(f'{arm} {recipe_name} s{seed}: verified weights; skip', flush=True)
            continue
        held = sorted((root / 'held' / arm).rglob('*.npz')) if (root / 'held' / arm).exists() else []
        assert not held, f'V4: no new training of {arm} once held outputs exist (any seed): {held[0]}'
        if state == 'broken':
            _move_aside(root, arm, seed, recipe_name)
        result[seed] = train_cohort(scale, s, inputs, graphs, manifests, images, encoder, recipe_name, seed, device,
                                    root, arm, smoke=smoke, max_epochs=max_epochs)
    return result


# --- output files -----------------------------------------------------------------------------------------------

def write_outputs(path, arrays, meta):
    """`path` (.npz) and a sibling .json: shape, dtype and sha256 of each array, writing-code hashes, then `meta`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **arrays)
    info = dict(arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha_array(v)) for k, v in arrays.items()},
                code_sha256=code_hashes(), **meta)
    ms_train.head.atomic_json(path.with_suffix('.json'), info)
    pending.replace(path)


def read_outputs(path):
    """Arrays and info of a file written by `write_outputs`, checked against the sibling record."""
    path = Path(path)
    info = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(info['arrays']), f'{path}: array names differ from the record'
        arrays = {k: z[k] for k in z.files}
    for key, record in info['arrays'].items():
        x = arrays[key]
        assert list(x.shape) == record['shape'] and str(x.dtype) == record['dtype'], f'{path}: {key} shape or dtype'
        assert sha_array(x) == record['sha256'], f'{path}: {key} sha256 differs from the record'
    return arrays, info


# --- inference --------------------------------------------------------------------------------------------------

@threadpool_limits.wrap(limits=8)
def infer(weights, aspect, width, x, device):
    """One saved best view (image side) on images `x`, `ct_d2.infer`'s way: batches of 512, float16 cast, checked."""
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[1] == width and x.dtype == np.float32 and np.isfinite(x).all()
    torch.set_num_threads(8)
    device = torch.device(device)
    _, views = build_model(rb_rows.recipe(rb_rows.REFERENCE)['base'], list(ASPECTS), device, width=width)
    view = views[aspect]
    view.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    view.eval()
    out = np.empty((len(x), OUT), dtype=np.float16)
    batch = ms_train.CONFIG['batch_size']
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xx = torch.from_numpy(np.ascontiguousarray(x[start:start + batch])).to(device)
            out[start:start + len(xx)] = view['img'](xx).cpu().numpy().astype(np.float16)
    assert np.isfinite(out).all()
    if len(out):
        assert np.max(np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1)) < 1e-3
    return out


def held_names():
    """The 12,281 held paintings in the CT's saved order (its list); the hash must be the rule's."""
    names, sha256 = ct_held.read_held_list(rb_paths.CT_OUT / 'held/held_paintings.json')
    assert sha256 == rb_paths.HELD_SHA256, 'held list hash differs from the rule\'s'
    return names, sha256


def require_regression(outputs, scale):
    """V1: held outputs of a new arm only after the scale's regression passed (stage 2 sample, stage 3a full)."""
    stage = {v: k for k, v in STAGES.items()}[scale]
    path = Path(outputs) / 'part2/regression' / f'{stage}.json'
    assert path.is_file(), f'V1: {path} missing; run --regress {stage} first'
    # the shared V1 rule (rb_score): sample = its stage 3, full = its stage 4 (which also reads verdict_stage3.json)
    rb_score.require_gate(Path(outputs), {'sample': 3, 'full': 4}[scale], [])


def ct_root(scale):
    return {'sample': rb_paths.CT_OUT, 'full': rb_paths.CT_FULL}[scale]


def _arm_weights(scale, arm, aspect, seed, root, smoke):
    """(path, config, encoder, recorded device) of an arm's saved best view; K1 is the CT's own file."""
    if arm == 'K1':
        info = ct_d2._weights_info(ct_root(scale), aspect, seed)
        assert info['config'].get('mode', 'sample') == scale
        return ct_d2._weight_path(ct_root(scale), aspect, seed), info['config'], 'qwen3vl_emb_2b', info['weights_sha256']
    encoder = ENCODER.get(arm, 'qwen3vl_emb_2b' if arm == REGRESSION_ARM else None)
    assert encoder, arm
    path = weights_path(root, arm, aspect, seed)
    info = json.loads(path.with_suffix('.json').read_text())
    assert ms_train.head.sha_file(path) == info['weights_sha256'], f'{path}: weights checksum mismatch'
    config = info['config']
    assert (config['aspect'], config['seed'], config['smoke']) == (aspect, seed, smoke)
    assert (config['rb']['scale'], config['rb']['encoder'], config['rb']['width']) == (
        scale, encoder, rb_encoders.WIDTH[encoder])
    return path, config, encoder, info['weights_sha256']


def infer_arm(scale, arm, seeds, kinds, outputs, device=None, *, smoke=False, root=None):
    """Held and/or ArtELingo-28 outputs of an arm's three best views for each seed.

    K1 (the CT's saved weights, `CT_OUT` or `CT_FULL`) takes ArtELingo-28 only, from its own Qwen store; the CT's held
    outputs are K1's. Files are `root/<kind>/<arm>/<aspect>__s{s}.npz` (root: part2/<scale>); an existing file must match.
    """
    seeds, kinds = list(seeds), tuple(kinds)
    assert scale in SCALES and seeds and len(set(seeds)) == len(seeds) and all(s in range(5) for s in seeds)
    assert kinds and set(kinds) <= {'held', 'a28'} and len(set(kinds)) == len(kinds)
    assert arm != 'K1' or kinds == ('a28',), 'K1 held outputs are the CT\'s own'
    outputs = _check_outputs(outputs, smoke)
    root = Path(root) if root is not None else outputs / 'part2' / scale
    if 'held' in kinds and arm != REGRESSION_ARM and not smoke:
        require_regression(outputs, scale)
    provider = {}
    for kind in kinds:
        if kind == 'held':
            names, sha256 = held_names()
            provider[kind] = sha256, names
        else:
            provider[kind] = rb_encoders.a28_sha256(), rb_encoders.a28_names()
    images = {}
    result = {}
    for seed in seeds:
        for aspect in ASPECTS:
            weights, config, encoder, weights_sha = _arm_weights(scale, arm, aspect, seed, root, smoke)
            recorded = torch.device(config['device'])
            assert device is None or torch.device(device) == recorded, 'inference device differs from training device'
            for kind in kinds:
                names_sha, names = provider[kind]
                if (kind, encoder) not in images:
                    images[kind, encoder] = (rb_encoders.full_images(encoder, names) if kind == 'held'
                                             else rb_encoders.a28_images(encoder))
                x = images[kind, encoder]
                width = rb_encoders.WIDTH[encoder]
                assert x.shape == (len(names), width)
                path = root / kind / arm / f'{aspect}__s{seed}.npz'
                meta = dict(kind=kind, arm=arm, scale=scale, aspect=aspect, seed=seed, encoder=encoder, rows=len(names),
                            width=OUT, dtype='float16', list_sha256=names_sha, weights_sha256=weights_sha,
                            image_sha256=sha_array(x), device=str(recorded), recipe=config.get('rb', {}).get('recipe'))
                if path.exists():
                    arrays, info = read_outputs(path)
                    assert all(info[k] == v for k, v in meta.items()), f'{path}: existing output differs from this run'
                    out = arrays['img']
                else:
                    out = infer(weights, aspect, width, x, recorded)
                    write_outputs(path, dict(img=out), dict(meta, sha256=sha_array(out)))
                    arrays, info = read_outputs(path)
                assert info['sha256'] == info['arrays']['img']['sha256'] == sha_array(out)
                result[kind, aspect, seed] = out
    return result


# --- regressions (stage 2 sample, stage 3a full) -----------------------------------------------------------------

def _bits(x):
    return np.ascontiguousarray(x).view(np.uint16 if x.dtype == np.float16 else np.uint8)


def regression_items(reg_root, ct, x_held, seed=0):
    """Compare our Qwen + R0-5 student (`reg_root`) with the CT's saved D2 under `ct`, bit for bit.

    Per aspect: best weights (every tensor), best epoch and dev loss, held output of our weights, and the held output of
    the CT's own weights through our inference path (`k1_replay`). `x_held`: the Qwen held images, float32 [12281, 2048].
    """
    items = {}
    for aspect in ASPECTS:
        parts = {}
        ours_pt = weights_path(reg_root, REGRESSION_ARM, aspect, seed)
        theirs_pt = ct_d2._weight_path(ct, aspect, seed)
        a, b = (torch.load(p, map_location='cpu', weights_only=True) for p in (ours_pt, theirs_pt))
        same = list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
        parts['weights'] = dict(
            **{'pass': bool(same)}, keys=list(a) == list(b),
            max_abs_diff=max((float((a[k].double() - b[k].double()).abs().max()) for k in a if k in b), default=None))
        oi, ti = (json.loads(p.with_suffix('.json').read_text()) for p in (ours_pt, theirs_pt))
        parts['best_epoch'] = {
            'pass': bool(oi['best_epoch'] == ti['best_epoch'] and oi['best_dev_loss'] == ti['best_dev_loss']),
            'ours': [oi['best_epoch'], oi['best_dev_loss']], 'ct': [ti['best_epoch'], ti['best_dev_loss']]}
        with np.load(ct / 'held/d2' / f'{aspect}__s{seed}.npz', allow_pickle=False) as z:
            assert z.files == ['img']
            theirs = z['img']
        ours, _ = read_outputs(reg_root / 'held' / REGRESSION_ARM / f'{aspect}__s{seed}.npz')
        ours = ours['img']
        replay = infer(theirs_pt, aspect, rb_encoders.WIDTH['qwen3vl_emb_2b'], x_held,
                       torch.device(ti['config']['device']))
        for name, got in (('held', ours), ('k1_replay_held', replay)):
            equal = got.shape == theirs.shape and got.dtype == theirs.dtype and np.array_equal(_bits(got), _bits(theirs))
            parts[name] = {'pass': bool(equal), 'rows': int(len(got)), 'ours_sha256': sha_array(got),
                           'ct_sha256': sha_array(theirs),
                           'max_abs_diff': float(np.abs(got.astype(np.float64) - theirs.astype(np.float64)).max())
                           if got.shape == theirs.shape else None}
        items[f'student/{aspect}'] = dict(**{'pass': all(v['pass'] for v in parts.values())}, **parts)
    return items


def regress(stage, device, outputs, *, smoke=False):
    """Train Qwen + R0-5 at seed 0 at the stage's scale, infer held, and write `part2/regression/<stage>.json` items."""
    assert stage in STAGES
    scale = STAGES[stage]
    outputs = _check_outputs(outputs, smoke)
    ct = ct_root(scale)
    recorded = ct_d2._weights_info(ct, ASPECTS[0], 0)['config']['device']
    assert str(torch.device(device)) == recorded, f'the CT trained on {recorded}; the regression must run on it'
    reg_root = outputs / 'part2/regression' / f'{stage}_run'
    train(scale, 'qwen3vl_emb_2b', rb_rows.REFERENCE, [0], device, outputs, arm=REGRESSION_ARM, root=reg_root)
    infer_arm(scale, REGRESSION_ARM, [0], ('held',), outputs, device, root=reg_root)
    names, sha256 = held_names()
    x_held = rb_encoders.full_images('qwen3vl_emb_2b', names)
    items = regression_items(reg_root, ct, x_held)
    items['student/held_list'] = {'pass': sha256 == rb_paths.HELD_SHA256, 'sha256': sha256}
    record = rb_score.write_regression_items(outputs / 'part2/regression' / f'{stage}.json', items)
    print(f'{stage}: pass={record["pass"]}; student items failing: '
          f'{[k for k, v in items.items() if not v["pass"]]}', flush=True)
    return record


# --- command line -----------------------------------------------------------------------------------------------

def picked_recipe(outputs):
    """The recipe of `part1/pick.json` (rule §4.1): P-KEEP is R0-5, else P-<recipe>."""
    pick = json.loads((Path(outputs) / 'part1/pick.json').read_text())
    name, outcome = pick['recipe'], pick['outcome']
    assert name in rb_rows.RECIPES, f'unknown recipe {name}'
    assert outcome == ('P-KEEP' if name == rb_rows.REFERENCE else f'P-{name}'), f'outcome {outcome} does not match {name}'
    return name


def main(argv=None):
    import argparse
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # the CT queue's setting, before any CUDA work
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
        p.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0; seed 0; at most 2 epochs')

    p = commands.add_parser('train', help='train the student of one encoder at one scale')
    p.add_argument('--scale', required=True, choices=SCALES)
    p.add_argument('--encoder', required=True, choices=list(rb_encoders.WIDTH))
    p.add_argument('--recipe', choices=list(rb_rows.RECIPES))
    p.add_argument('--pick', action='store_true', help='the recipe of OUTPUTS/part1/pick.json')
    p.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-epochs', type=int)
    common(p)
    p = commands.add_parser('infer', help='held and/or ArtELingo-28 outputs from saved best views (K1: the CT\'s)')
    p.add_argument('--scale', required=True, choices=SCALES)
    p.add_argument('--arm', required=True, choices=['K1', 'K2', 'K3', 'K4', 'K5'])
    p.add_argument('--kinds', nargs='+', choices=['held', 'a28'], default=['held', 'a28'])
    p.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    p.add_argument('--device', help='default: the device the weights were trained on')
    common(p)
    p = commands.add_parser('regress', help='Qwen + R0-5 seed 0 against the CT: stage2 (sample) or stage3a (full)')
    p.add_argument('stage', choices=list(STAGES))
    p.add_argument('--device', default='cuda')
    common(p)
    args = parser.parse_args(argv)
    if args.command == 'train':
        if (args.recipe is None) == (not args.pick):
            parser.error('give exactly one of --recipe and --pick')
        recipe = args.recipe or picked_recipe(args.outputs)
        seeds = [0] if args.smoke else args.seeds
        return train(args.scale, args.encoder, recipe, seeds, args.device, args.outputs, smoke=args.smoke,
                     max_epochs=args.max_epochs)
    if args.command == 'infer':
        seeds = [0] if args.smoke else args.seeds
        return infer_arm(args.scale, args.arm, seeds, args.kinds, args.outputs, args.device, smoke=args.smoke)
    return regress(args.stage, args.device, args.outputs, smoke=args.smoke)


if __name__ == '__main__':
    main()
