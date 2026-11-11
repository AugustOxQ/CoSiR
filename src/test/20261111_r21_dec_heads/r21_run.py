"""r21 runner, train part (ticket 03): an r21 arm trained on c*'s exact chain.

c* is the recipe sweep's configuration `k11=40`. `rs_full.train` (`20261027_recipe_sweep/rs_full.py`, frozen, imported,
never edited) trains it by: `rs_knobs.patches` (the 40-neighbour graphs), `arm_alias`, `load_data`, the rb block, and
`rg_full.train_cohort` (which ends in `ct2_d2.run`). `chain` below repeats exactly those steps. The only difference of
an r21 arm: `ct2_d2.run` is replaced, through its module attribute, by `r21_train.run` carrying the arm's settings, so
every other patch of the chain (loss, model, config blocks, weight saver, k11's graphs) still applies. DEC off
(`dec-off`) runs `r21_train.run` with its hooks idle; `settings=None` runs the frozen `ct2_d2.run` (gate 0b's reference).

Guards that replace rs_full's sweep guards (stage 3 verdict, V4a):
  * the arm name parses (`parse_arm`);
  * outputs go only under the r21 tree `OUT_ROOT` (a full run) or under `/tmp/claude-0` (a smoke);
  * r10's tree is only read: its listing (path, size, mtime) is taken before and after the run and must be equal;
  * `CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA (set before torch is imported; asserted);
  * the frozen modules equal their files at `rs_paths.FROZEN_COMMIT` (rs_full.require_frozen_code);
  * resume: a cohort with verified weights is skipped only if its run record holds the same arm settings, smoke flag,
    frozen modules and code SHA-256 (rs_full.code_problem); otherwise the run stops and names the difference.

Layout (index, "Output tree"): `<root>/weights/<arm>/{aspect}__s{seed}.pt|json`, `<root>/train/<arm>/R6-5/...` (validation
exports) and `<root>/train/<arm>/run__s{seed}.json` (the run record, below), `<root> = <outputs>/full`; the teacher's
cluster cache is `<outputs>/teacher_clusters/{aspect}.npz`.

Run record: arm, settings, seed, smoke, max_epochs, device, status, the knobs and graphs of c*'s configuration, SHA-256 of
the frozen and the other code, git commit and dirty flag, seconds, and per head `heads.<aspect>` = stop_reason,
stop_epoch, dec_start_epoch (None with DEC off), best_epoch, best_dev_loss, epochs_run: read from the validation sidecar's
history. DEC off has no DEC stop rule: its stop_reason is c*'s ('patience', or 'max_epochs' at the cap).

CLI: r21_run.py train --arm NAME [NAME ...] --seeds 0 1 2 3 4 --device cuda [--smoke] [--max-epochs N] [--outputs DIR]

Held part (ticket 04): `held` writes the test exports of an arm's saved views, `<root>/held/<arm>/{aspect}__s{seed}.npz|json`,
by c*'s own inference (`rg_full.infer_held`'s steps: the CT's held images, `rb_student.infer` on the saved view, VC3, one
rerun on a failure, c*'s sidecar fields), without r10's sweep gates (stage 3 verdict, V4a, five seeds at once). The inference
device is the training device in the weights' record (as c*); `--device` may name another only with `--allow-device-change`
(a CPU check of CUDA-trained weights), which adds `infer_device` to the sidecar. `cstar` as the arm reads r10's saved c*
weights (`full/weights/k11=40`, read-only) and writes the exports as arm `k11=40`; only into a smoke folder.
`teacher` writes the teacher-cluster cache once (D12) through `teacher_data`; a later call verifies it. CPU.

CLI: r21_run.py held --arm NAME|cstar [...] --seeds 0 .. [--device D --allow-device-change] [--outputs DIR]
     r21_run.py teacher [--outputs DIR]
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before torch initialises CUDA (CT queue, rs_full)

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20261027_recipe_sweep'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import rs_paths  # noqa: E402  (puts the older rounds' folders on sys.path)
import r21_train  # noqa: E402
from r21_arms import arm_name, parse_arm  # noqa: E402,F401  (re-exported: index, "Shared interfaces")
import rs_full  # noqa: E402
import rs_knobs  # noqa: E402
import rs_sheet as S  # noqa: E402
import rs_train  # noqa: E402
import ct2_d2  # noqa: E402
import ct2_data  # noqa: E402
import r21_dec  # noqa: E402
import rb_encoders  # noqa: E402
import rb_rows  # noqa: E402
import rb_student  # noqa: E402
import rg_full  # noqa: E402
from ab_paths import ms_train  # noqa: E402

ASPECTS = tuple(rs_paths.ASPECTS)
CONFIG_NAME = 'k11=40'                                       # c*
OUT_ROOT = Path('/data/SSD2/cosir_r21/20261111_dec_heads')   # the r21 tree; its `full` folder is the index's R
R = OUT_ROOT / 'full'
R10_OUTPUTS = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')   # r10's tree: only read
SMOKE_ROOT = Path('/tmp/claude-0')
CLASS_COUNT = dict(emotion=9, style=27, genre=10)            # spec §3.4; = ns_paths.CLASS_COUNT for ArtELingo
SEEDS = rg_full.SEEDS
TEACHER_CACHE = 'teacher_clusters'


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------- guards
def check_outputs(outputs, smoke):
    """(outputs, root): a full run writes only under the r21 tree, a smoke only under /tmp/claude-0 (never into r10's)."""
    outputs = Path(outputs).resolve()
    if smoke:
        assert outputs.is_relative_to(SMOKE_ROOT), f'--smoke outputs must be under {SMOKE_ROOT}: {outputs}'
    else:
        assert outputs == OUT_ROOT.resolve(), f'a full run writes only under {OUT_ROOT}: {outputs}'
    assert not outputs.is_relative_to(R10_OUTPUTS.resolve()) and not R10_OUTPUTS.resolve().is_relative_to(outputs), \
        f'outputs and r10\'s tree overlap: {outputs}'
    rs_paths.check_outputs(outputs, smoke)  # also: not inside a frozen input of the earlier rounds
    return outputs, outputs / 'full'


def tree_listing(top=R10_OUTPUTS):
    """{relative path: (size, mtime_ns)} of every file under `top`: r10's tree must be the same after a run."""
    top = Path(top)
    return {str(p.relative_to(top)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(top.rglob('*')) if p.is_file()}


def git_state():
    """(HEAD, dirty): dirty when a Python file of this round, the recipe sweep or 12-27 is modified or untracked."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=rs_paths.ROUND, capture_output=True, text=True, check=True).stdout.strip()
    dirty = git('status', '--porcelain', '--', ':/src/test/20261111_r21_dec_heads/*.py',
                ':/src/test/20261027_recipe_sweep/*.py', ':/src/test/20261027_relational_genre/*.py')
    return git('rev-parse', 'HEAD'), bool(dirty)


# ----------------------------------------------------------------------------------------------- teacher clusters
def _sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def cache_path(cache_dir, aspect):
    return Path(cache_dir) / f'{aspect}.npz'


def teacher_data(s, graphs, aspects, cache_dir):
    """The teacher form's `dec_data`: per aspect `k` (class count), `feats` (float32 [n_fit, 2048], unit rows: the
    instructed Qwen image of the fit paintings, in `graphs[aspect]['fit_node_ids']` order, the node ids being the
    extraction's painting positions, C6; the IDs are also given as `fit_node_ids`) and `centres` (D12: `r21_dec.teacher_clusters`, k-means with k = class count,
    10 inits, state 0). The clusters are cached in `cache_path`: written on first use (centres, labels, the input's
    SHA-256, k), verified on every later read (the SHA of the rows now built, k, shapes) and never silently rebuilt."""
    out = {}
    for aspect in aspects:
        teacher = ct2_data.instructed_images(s, ms_train.head.ASPECTS[aspect])
        assert teacher.shape == (len(s['first']), 2048) and teacher.dtype == np.float32 and np.isfinite(teacher).all()
        assert _sha_array(teacher) == s['data_sha256'][f'{aspect}.img'], f'{aspect}: teacher differs from the CT\'s'
        ids = np.asarray(graphs[aspect]['fit_node_ids'])
        assert set(ids.tolist()) == set(np.asarray(s['fit']).tolist()) and len(ids) == len(s['fit'])
        feats = np.ascontiguousarray(teacher[ids])
        del teacher
        k = CLASS_COUNT[aspect]
        input_sha = _sha_array(feats)
        path = cache_path(cache_dir, aspect)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                assert set(z.files) == {'centres', 'labels', 'input_sha256', 'k'}, f'{path}: keys {z.files}'
                centres, labels = z['centres'], z['labels']
                assert str(z['input_sha256']) == input_sha, f'{path}: built from other rows than the fit paintings now'
                assert int(z['k']) == k and centres.shape == (k, 2048) and centres.dtype == np.float32, path
                assert labels.shape == (len(feats),) and labels.min() >= 0 and labels.max() < k, path
        else:
            centres, labels = r21_dec.teacher_clusters(feats, k)
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_name(path.stem + '.pending.npz')
            np.savez(pending, centres=centres, labels=labels, input_sha256=np.array(input_sha), k=np.array(k))
            pending.replace(path)
        out[aspect] = dict(k=k, feats=feats, centres=centres, fit_node_ids=ids.copy())   # C6: the rows' extraction IDs
    return out


def dec_data_for(settings, s, graphs, cache_dir):
    """`r21_train.run`'s dec_data: the class count per aspect; the teacher form adds the rows and the centres."""
    if settings is None or settings['form'] == 'off':
        return None
    if settings['form'] == 'teacher':
        return teacher_data(s, graphs, ASPECTS, cache_dir)
    return {a: dict(k=CLASS_COUNT[a]) for a in ASPECTS}


# ------------------------------------------------------------------------------------------------------- the chain
def load_chain(stack, outputs_for_graphs=R10_OUTPUTS):
    """c*'s steps before `rg_full.train_cohort` (rs_full.train): the knob patches of `k11=40` (graphs and gates are read
    from r10's tree), the data and the rb block. Returns the context dict."""
    cfg = S.parse(CONFIG_NAME)
    knobs = rs_knobs.patches(cfg, stack, scale='full', outputs=Path(outputs_for_graphs))
    s, inputs, graphs, manifests, images = rs_full.load_data()
    rb = rb_student._rb_block(rg_full.SCALE, rg_full.ENCODER, rg_full.RECIPE, rb_rows.recipe(rg_full.RECIPE), images)
    return dict(cfg=cfg, knobs=knobs, s=s, inputs=inputs, graphs=graphs, manifests=manifests, images=images, rb=rb)


R21_CODE_FILES = ('r21_dec.py', 'r21_train.py', 'r21_run.py')   # what decides an r21 arm's weights besides the frozen chain


def r21_code_sha256():
    return {name: sha_file(HERE / name) for name in R21_CODE_FILES}


def with_arm(original, settings):
    """`rg_full._with_blocks` plus two config fields of an r21 run: `dec` (the arm's settings) and `r21_code_sha256`. The
    trainer's skip ("verified identical outputs") compares configs, so only the same settings and code can match."""
    def with_blocks(make_config, rb, rg):
        inner = original(make_config, rb, rg)

        def wrapped(*args, **kwargs):
            config = inner(*args, **kwargs)
            assert 'dec' not in config and 'r21_code_sha256' not in config
            config['dec'], config['r21_code_sha256'] = copy.deepcopy(settings), r21_code_sha256()
            return config
        return wrapped
    return with_blocks


def run_cohort(ctx, name, settings, seed, device, root, *, smoke, max_epochs=None, dec_data=None):
    """`rg_full.train_cohort` for the run `name` (an arm of `rg_variants` with c*'s switches): with `settings` (an arm's
    dict) `ct2_d2.run` is `r21_train.run` carrying them, and every sidecar config also holds `dec` and the r21 code
    hashes; with None it is the frozen `ct2_d2.run` and c*'s configs."""
    with ExitStack() as stack:
        stack.enter_context(rs_full.arm_alias(name))
        if settings is not None:
            stack.enter_context(patch.object(ct2_d2, 'run', partial(r21_train.run, dec=settings, dec_data=dec_data)))
            stack.enter_context(patch.object(rg_full, '_with_blocks', with_arm(rg_full._with_blocks, settings)))
        return rg_full.train_cohort(ctx['s'], ctx['inputs'], ctx['graphs'], ctx['manifests'], ctx['images'], name, seed,
                                    torch.device(device), root, rb=ctx['rb'], smoke=smoke, max_epochs=max_epochs)


# ------------------------------------------------------------------------------------------------------- records
def record_path(root, name, seed):
    return Path(root) / 'train' / name / f'run__s{seed}.json'


def head_records(root, name, seed, settings):
    """Per head, from the validation sidecar's history: stop_reason, stop_epoch, dec_start_epoch, best epoch and loss.
    Asserts what the trainer promises (r21_train docstring): with DEC on every head reaches DEC and its last entry says why
    it stopped; the validation export exists and its hash is the sidecar's."""
    heads = {}
    for aspect in ASPECTS:
        info = json.loads(rb_student.weights_path(root, name, aspect, seed).with_suffix('.json').read_text())
        config = info['config']
        npz = ms_train.output_path(Path(root) / 'train' / name, rg_full.RECIPE, aspect, config['graph_row'], seed)
        side = json.loads(npz.with_suffix('.json').read_text())
        assert side['output_sha256'] == info['output_sha256'] == sha_file(npz), f'{npz}: file hash differs from its sidecar'
        history = side['history']
        assert history and len(history) == side['epochs_run'], npz
        last = history[-1]
        assert last['epoch'] == len(history) and [h['epoch'] for h in history] == list(range(1, len(history) + 1)), npz
        if settings['form'] == 'off':
            assert not any('dec' in h for h in history), f'{npz}: DEC entries in a DEC-off run'
            cap = config['max_epochs']
            reason = 'max_epochs' if last['epoch'] >= cap else 'patience'
            if reason == 'patience':
                assert last['epoch'] - side['best_epoch'] >= ms_train.CONFIG['patience'], f'{npz}: stopped early without patience'
            start = None
        else:
            assert all('dec' in h for h in history), f'{npz}: a history entry without DEC stats'
            assert last['dec']['phase'] == 'dec' and 'stop_reason' in last['dec'], f'{npz}: the head never reached DEC or has no stop reason'
            reason = last['dec']['stop_reason']
            first_dec = next(h for h in history if h['dec']['phase'] == 'dec')
            start = first_dec['dec']['start_epoch']
            assert start == first_dec['epoch'] == settings['warmup'] + 1, (npz, start, settings['warmup'])
            assert all(h['dec']['phase'] == 'warmup' for h in history[:start - 1])
            assert all(np.isfinite(h['train_loss']) and np.isfinite(h['dev_loss']) for h in history), npz
        heads[aspect] = dict(stop_reason=reason, stop_epoch=last['epoch'], dec_start_epoch=start,
                             best_epoch=side['best_epoch'], best_dev_loss=side['best_dev_loss'], epochs_run=side['epochs_run'],
                             output_sha256=side['output_sha256'], weights_sha256=info['weights_sha256'])
    return heads


def check_resumed(root, name, seed, settings, smoke, max_epochs, shas):
    """A cohort with verified weights is skipped only if its record is this arm's, trained by this code."""
    path = record_path(root, name, seed)
    assert path.is_file(), f'{path}: weights without a run record; move them aside'
    record = json.loads(path.read_text())
    assert record['status'] == 'ok', f'{path}: status {record["status"]}'
    assert (record['arm'], record['settings'], record['seed'], record['smoke'], record['max_epochs']) == \
        (name, settings, seed, smoke, max_epochs), f'{path}: settings mismatch'
    assert record['frozen_sha256'] == shas, f'{path}: frozen modules changed'
    problem = rs_full.code_problem(record['code_sha256'])
    assert problem is None, f'{path}: code changed ({problem}); move it aside and rerun'
    assert record['heads'] == head_records(root, name, seed, settings), f'{path}: heads differ from the exports'
    return record


# --------------------------------------------------------------------------------------------------------- train
@threadpool_limits.wrap(limits=8)
def train(arms, seeds, device, outputs=OUT_ROOT, *, smoke=False, max_epochs=None):
    """Train each arm of `arms` for each seed on c*'s chain. Returns {arm: {seed: run record}}."""
    arms, seeds = list(arms), list(seeds)
    settings = {a: parse_arm(a) for a in arms}
    assert arms and len(set(arms)) == len(arms), arms
    assert seeds and len(set(seeds)) == len(seeds) and all(x in SEEDS for x in seeds), f'seeds: {seeds}'
    assert not smoke or (max_epochs or 2) <= 2, '--smoke: at most 2 epochs'
    assert max_epochs is None or smoke, 'a full run trains to the CT\'s 200 epochs and its patience'
    outputs, root = check_outputs(outputs, smoke)
    device = torch.device(device)
    if device.type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (CT queue)'
    rs_paths.check_imports()
    shas = rs_full.require_frozen_code()                   # before any data is read
    before = tree_listing()
    for arm in arms:
        for seed in seeds:
            assert not rg_full.failed_path(root, arm, seed).exists(), \
                f'{arm} s{seed} stopped on a non-finite loss ({rg_full.failed_path(root, arm, seed)}); no rerun'
    torch.set_num_threads(8)
    result = {}
    with ExitStack() as stack:
        ctx = load_chain(stack)
        dec_data = {a: dec_data_for(settings[a], ctx['s'], ctx['graphs'], outputs / TEACHER_CACHE) for a in arms}
        for arm in arms:
            result[arm] = {}
            for seed in seeds:
                with ExitStack() as inner:
                    inner.enter_context(rs_full.arm_alias(arm))   # the state check below reads V.ARMS[arm]
                    rg = rg_full.rg_block(arm, seed, len(ctx['s']['fit']))
                    state = rg_full._cohort_state(root, arm, seed, ctx['rb'], rg)
                if state == 'done':
                    result[arm][seed] = check_resumed(root, arm, seed, settings[arm], smoke, max_epochs, shas)
                    print(f'{arm} s{seed}: verified weights; skip', flush=True)
                    continue
                if state == 'broken':
                    rb_student._move_aside(root, arm, seed, rg_full.RECIPE)
                start = time.monotonic()
                outcome = run_cohort(ctx, arm, settings[arm], seed, device, root, smoke=smoke, max_epochs=max_epochs,
                                     dec_data=dec_data[arm])
                commit, dirty = git_state()
                record = dict(arm=arm, settings=settings[arm], seed=seed, smoke=smoke, max_epochs=max_epochs,
                              device=str(device), status='ok', cause=None, config=rs_full.jsonable(ctx['cfg']),
                              config_name=CONFIG_NAME, knobs=ctx['knobs'], graphs=rs_full.graph_choice(ctx['cfg']),
                              frozen_commit=rs_paths.FROZEN_COMMIT, frozen_sha256=shas, code_sha256=rs_full.code_sha256(),
                              code_commit=commit, code_dirty=dirty, host=socket.gethostname(),
                              teacher_cache=({a: sha_file(cache_path(outputs / TEACHER_CACHE, a)) for a in ASPECTS}
                                             if settings[arm]['form'] == 'teacher' else None))
                if outcome.get('status') == 'failed':
                    record.update(status='failed', cause=outcome['cause'], heads=None)
                else:
                    record['heads'] = head_records(root, arm, seed, settings[arm])
                record['seconds'] = time.monotonic() - start
                ms_train.head.atomic_json(record_path(root, arm, seed), record)
                result[arm][seed] = record
                print(f'{arm} s{seed}: {record["status"]}, {record["seconds"]:.0f} s', flush=True)
    assert tree_listing() == before, 'r10\'s tree changed during the run (it is only read)'
    return result


# ---------------------------------------------------------------------------------------------------------- held
CSTAR_ARM = 'k11=40'    # c*'s folder name in r10's tree; also the `arm` of its sidecars


def _outputs_for(outputs):
    """(outputs, root, smoke): the r21 tree (a full run) or a folder under /tmp/claude-0 (a smoke), as `check_outputs`."""
    smoke = Path(outputs).resolve() != OUT_ROOT.resolve()
    return (*check_outputs(outputs, smoke), smoke)


def held_arm_weights(wroot, arm, aspect, seed, smoke):
    """(path, config, weights sha) of the saved view, checked as `rg_full._arm_weights` (inside `arm_alias(arm)`)."""
    return rg_full._arm_weights(wroot, arm, aspect, seed, smoke)


@threadpool_limits.wrap(limits=8)
def held(arms, seeds, device=None, outputs=OUT_ROOT, *, allow_device_change=False):
    """Held exports of each arm of `arms` (r21 arm names, or 'cstar' for r10's saved c*, smoke folders only) for each seed.
    Returns {(arm, seed): {aspect: float16 array}}. An existing export is kept if its sidecar matches this run (weights
    SHA-256 included) and passes VC3; a changed one stops the run."""
    arms, seeds = list(arms), list(seeds)
    assert arms and len(set(arms)) == len(arms) and seeds and len(set(seeds)) == len(seeds) and all(x in SEEDS for x in seeds)
    outputs, root, smoke = _outputs_for(outputs)
    folder = {a: (CSTAR_ARM if a == 'cstar' else a) for a in arms}
    for a in arms:
        assert a == 'cstar' or parse_arm(a)
    assert 'cstar' not in arms or smoke, 'cstar (r10\'s weights) only into a smoke folder; the r21 tree holds r21 arms'
    wroot = {a: (R10_OUTPUTS / 'full' if a == 'cstar' else root) for a in arms}
    rs_paths.check_imports()
    rs_full.require_frozen_code()
    before = tree_listing()
    for a in arms:
        for seed in seeds:
            rg_full._refuse_failed(root, folder[a], [seed])
            if a != 'cstar':   # the arm's training run must have finished well
                rec = json.loads(record_path(root, a, seed).read_text())
                assert rec['status'] == 'ok' and rec['smoke'] == smoke, f'{a} s{seed}: run record {rec["status"]}, smoke {rec["smoke"]}'
    torch.set_num_threads(8)
    names, names_sha = rb_student.held_names()
    x = rb_encoders.full_images(rg_full.ENCODER, names)
    assert x.shape == (len(names), rb_encoders.WIDTH[rg_full.ENCODER]) == (rg_full.rg_paths.N_HELD, 2048) and x.dtype == np.float32
    assert np.isfinite(x).all() and names_sha == rs_paths.HELD_SHA256
    image_sha = rb_student.sha_array(x)
    result = {}
    for a in arms:
        name = folder[a]
        with rs_full.arm_alias(name):
            switches = rg_full.V.switches(name)
            weights = {(seed, aspect): held_arm_weights(wroot[a], name, aspect, seed, smoke and a != 'cstar') for seed in seeds for aspect in ASPECTS}
        for seed in seeds:
            result[a, seed] = {}
            for aspect in ASPECTS:
                w_path, config, w_sha = weights[seed, aspect]
                recorded = torch.device(config['device'])
                dev = recorded if device is None else torch.device(device)
                assert dev == recorded or allow_device_change, \
                    f'{w_path}: trained on {recorded}, inference on {dev}: pass --allow-device-change for a check'
                path = root / 'held' / name / f'{aspect}__s{seed}.npz'
                meta = dict(kind='held', arm=name, scale=rg_full.SCALE, aspect=aspect, seed=seed, encoder=rg_full.ENCODER,
                            rows=len(names), width=2048, dtype='float16', list_sha256=names_sha, weights_sha256=w_sha,
                            image_sha256=image_sha, device=str(recorded), recipe=rg_full.RECIPE,
                            switches=switches)
                if path.exists():
                    arrays, info = rb_student.read_outputs(path)
                    assert all(info[k] == v for k, v in meta.items()), f'{path}: existing output differs from this run'
                    out = arrays['img']
                    problem = rg_full.vc3_problem(out)
                    assert problem is None, f'{path}: VC3: {problem}'
                    print(f'{name} {aspect} s{seed}: verified export; skip', flush=True)
                else:
                    problems = []
                    for attempt in (1, 2):
                        try:
                            out = rb_student.infer(w_path, aspect, 2048, x, dev)
                            problem = rg_full.vc3_problem(out)
                        except AssertionError as error:
                            problem = f'infer: {error}'
                        if problem is None:
                            break
                        problems.append(problem)
                        print(f'{path}: VC3 failure ({problem}); {"rerun once" if attempt == 1 else "stop"}', flush=True)
                    else:
                        ms_train.head.atomic_json(path.parent / f'failed__{aspect}__s{seed}.json',
                                                  dict(arm=name, aspect=aspect, seed=seed, failure_kind='vc3', problems=problems))
                        raise AssertionError(f'{path}: VC3 failed twice: {problems}')
                    extra = dict(infer_device=str(dev)) if dev != recorded else {}
                    rb_student.write_outputs(path, dict(img=out), dict(meta, sha256=rb_student.sha_array(out),
                                             vc3_attempts=len(problems) + 1, vc3_problems=problems, **extra))
                    arrays, info = rb_student.read_outputs(path)
                    print(f'{name} {aspect} s{seed}: written', flush=True)
                assert info['sha256'] == info['arrays']['img']['sha256'] == rb_student.sha_array(out)
                result[a, seed][aspect] = out
    assert tree_listing() == before, 'r10\'s tree changed during the run (it is only read)'
    return result


# ------------------------------------------------------------------------------------------------------- teacher
def teacher(outputs=OUT_ROOT):
    """Write (first call) or verify (later calls) the teacher-cluster cache of the three aspects. CPU."""
    outputs, _, _ = _outputs_for(outputs)
    rs_paths.check_imports()
    rs_full.require_frozen_code()
    before = tree_listing()
    with ExitStack() as stack:
        ctx = load_chain(stack)
        data = teacher_data(ctx['s'], ctx['graphs'], ASPECTS, outputs / TEACHER_CACHE)
    assert tree_listing() == before, 'r10\'s tree changed (it is only read)'
    return {a: dict(k=d['k'], rows=len(d['feats']), path=str(cache_path(outputs / TEACHER_CACHE, a)),
                    sha256=sha_file(cache_path(outputs / TEACHER_CACHE, a))) for a, d in data.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='train r21 arms on c*\'s chain')
    t.add_argument('--arm', nargs='+', required=True)
    t.add_argument('--seeds', nargs='+', type=int, required=True)
    t.add_argument('--device', required=True)
    t.add_argument('--smoke', action='store_true')
    t.add_argument('--max-epochs', type=int, default=None, help='smoke only (at most 2)')
    t.add_argument('--outputs', type=Path, default=OUT_ROOT)
    h = sub.add_parser('held', help='held (test) exports of an arm\'s saved views, c*\'s inference')
    h.add_argument('--arm', nargs='+', required=True, help="r21 arm names, or 'cstar' (smoke folders only)")
    h.add_argument('--seeds', nargs='+', type=int, required=True)
    h.add_argument('--device', default=None, help='default: the training device in the weights\' record')
    h.add_argument('--allow-device-change', action='store_true')
    h.add_argument('--outputs', type=Path, default=OUT_ROOT)
    c = sub.add_parser('teacher', help='write or verify the teacher-cluster cache (CPU)')
    c.add_argument('--outputs', type=Path, default=OUT_ROOT)
    args = p.parse_args(argv)
    if args.command == 'held':
        res = held(args.arm, args.seeds, args.device, args.outputs, allow_device_change=args.allow_device_change)
        print(json.dumps({f'{a} s{sd}': 'ok' for a, sd in res}))
        return
    if args.command == 'teacher':
        print(json.dumps(teacher(args.outputs), indent=1))
        return
    result = train(args.arm, args.seeds, args.device, args.outputs, smoke=args.smoke, max_epochs=args.max_epochs)
    print(json.dumps({a: {str(sd): r['status'] for sd, r in v.items()} for a, v in result.items()}))


if __name__ == '__main__':
    main()
