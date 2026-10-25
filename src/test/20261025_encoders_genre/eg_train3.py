"""Stage iii trainer (rule §2.3, §4.3): R6-5 with one change to the genre head's loss, on the frozen sample.

Training is 12-24's part 1 trainer: `ms_train.run` (the model sheet's loop) with the recipe R6-5 (`rb_rows`), the
same data, graphs and teacher images. Three things are patched around it for the length of one run, and each leaves
every random draw and every operation of the emotion and style paths as they were:
  * `build_model`: the sheet's factory, then (GA only) the adaptor, appended after the model's modules, so it comes
    last in the optimizer and inside the genre view (its state is saved and restored with the genre best state);
  * `head_loss`: `ab_train.switched_head_loss` with the genre-only changes (below); with every switch off, and for
    emotion and style always, it calls `ab_train.switched_head_loss` itself;
  * `save_output`: writes the exports of the aspect's best state (image head outputs on P and on the old 5,000).
The genre changes (switches `m`, `lam`, `w` of `eg_variants.switches`), all on the same head output `z` and the same
batch ids, beside the direct teacher term:
  GT-m  direct term weight 0.25 * m;
  GR-l  plus l * relational distillation of `z` toward the teacher rows;
  GA-w  plus w * `rb_probe.loss_fn(adaptor(z), standardized teacher rows)`, the standardizer fit on the 13,400 fit
        paintings' teacher rows, the adaptor Linear 2048-2048, GELU, Linear 2048-2048 from its own generator.
The same function computes the early stopping (dev) loss, so the new terms are in it.

Files in `eg_paths.iii_run(outputs, variant, seed)` (the frozen loop's own folder stays in `trainer/`, empty):
  `<aspect>.pt` (best state; the genre state of GA also holds `adaptor.*`), `<aspect>.json` (best epoch, history,
  array hashes), `<aspect>__P.npz` and `<aspect>__old5000.npz` (key `img`, float16 [n, 2048]),
  `genre__P_adaptor.npz` (GA, key `adaptor`, float16 [29232, 2048]), `run.json`.
Commands: `train`, `regress vg1`, `parity` (see `main`).
"""
import argparse
import json
import math
import os
import subprocess
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import eg_paths
import eg_variants as V
import ab_rows
import ab_train
import rb_part1
import rb_probe
import rb_rows
from ab_paths import ms_train

BASE = V.BASE                                   # 'R6-5'
GENRE = 'genre'
ASPECTS = tuple(ab_rows.ASPECTS)
assert ASPECTS == tuple(eg_paths.ASPECTS)
N_P = 29232
N_OLD = 5000
WIDTH = 2048
K4_WEIGHTS = eg_paths.RB_PART2 / 'sample/weights/K4'     # 12-24's K4 students are R6-5 on the same data (rule §4.3)


class NonFiniteLoss(Exception):
    """A train or dev loss that is not finite; the run stops with status failed."""


# ------------------------------------------------------------------ the genre loss
class GenreLoss:
    """The switches of one run (m, lam, w of `eg_variants.switches`) and what they need: the adaptor (set by the
    model factory) and the target standardizer (`TorchStd`, set by the cohort driver)."""

    def __init__(self, switches, seed, std=None):
        self.m, self.lam, self.w = switches['m'], switches['lam'], switches['w']
        assert self.m > 0 and self.lam >= 0 and self.w >= 0, switches
        self.seed, self.std, self.adaptor = seed, std, None

    @property
    def off(self):
        return self.m == 1 and self.lam == 0 and self.w == 0


class TorchStd:
    """`rb_probe.Standardizer.fwd` on the training device: float64 (x - mean) / std, then float32 (same results)."""

    def __init__(self, std, device):
        self.mean = torch.as_tensor(std.mean, dtype=torch.float64, device=device)
        self.std = torch.as_tensor(std.std, dtype=torch.float64, device=device)

    def fwd(self, x):
        return ((x.to(torch.float64) - self.mean) / self.std).to(torch.float32)


def switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher=None, extras=None,
                       include_mask_penalty=True, *, switches=None, genre=None):
    """`ab_train.switched_head_loss` plus the genre changes. Everything but a genre head with a switch on is
    the original function, called as it is."""
    if genre is None or genre.off or aspect != GENRE:
        return ab_train.switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher, extras,
                                           include_mask_penalty, switches=switches)
    # The R6-5 form (the only one with changes): both endpoints, buddy terms only, the direct teacher term.
    assert switches == rb_rows.RECIPES[BASE]['switches'], switches
    assert row['form'] == 'a' and row['graph_terms'] and extras is None and row['distill'] == 'direct'
    a, b = torch.as_tensor(pairs, device=img.device).unbind(1)
    ids = torch.cat((a, b))
    z = models['img'](img[ids])
    ia, ib = z.chunk(2)
    w = torch.as_tensor(weights, device=img.device)
    ca, cb = models['cap'](cap[ids]).chunk(2)
    sym = ms_train.symmetric
    terms = [(sym(ia, ib, w) + sym(ia, cb, w) + sym(ca, ib, w) + sym(ca, cb, w)) / 4]
    weight = row['distill_weight'] if genre.m == 1 else row['distill_weight'] * genre.m
    terms.append(weight * ms_train.distill_loss(z, teacher[ids], ids, 'direct'))
    if genre.lam:
        terms.append(genre.lam * ms_train.distill_loss(z, teacher[ids], ids, 'relational'))
    if genre.w:
        assert genre.adaptor is not None and genre.std is not None
        terms.append(genre.w * rb_probe.loss_fn(genre.adaptor(z), genre.std.fwd(teacher[ids])))
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def checked_head_loss(*args, genre=None, **kwargs):
    """The switched loss, stopping the run on a value that is not finite (train and dev calls alike)."""
    loss = switched_head_loss(*args, genre=genre, **kwargs)
    if not bool(torch.isfinite(loss)):
        raise NonFiniteLoss(kwargs['aspect'] if 'aspect' in kwargs else args[6])
    return loss


# ------------------------------------------------------------------ the adaptor
def make_adaptor(seed, device):
    """Linear 2048-2048, GELU, Linear 2048-2048, initialised as nn.Linear does (uniform, bound 1 / sqrt(fan_in)) from
    its own generator. The layers are created under a forked random state, so the global streams are as before."""
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        net = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
    with torch.no_grad():
        for layer in (net[0], net[2]):
            bound = 1 / math.sqrt(layer.in_features)
            layer.weight.uniform_(-bound, bound, generator=generator)
            layer.bias.uniform_(-bound, bound, generator=generator)
    return net.to(device)


def build_model_with(genre, original):
    """`original` (the sheet's factory), then the adaptor of a GA run appended to the models and the genre view."""
    def build(row, aspects, device):
        models, views = original(row, aspects, device)
        if genre.w:
            assert GENRE in views and genre.adaptor is None
            genre.adaptor = make_adaptor(genre.seed, device)
            models['adaptor'] = genre.adaptor            # after img and cap: last in the optimizer's parameters
            views[GENRE]['adaptor'] = genre.adaptor      # in the genre view: part of its saved best state
        return models, views
    return build


# ------------------------------------------------------------------ exports
def bits_equal(a, b):
    """Same dtype, shape and bytes (so signed zeros and NaN payloads count)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def vg3_problem(x, n_rows):
    """VG3 on one export: None, or the first problem (rows, dtype, finite, unit norm). 12-24's VP2 check."""
    import rb_pick
    return rb_pick.vp2_problem(x, n_rows)


def infer_img(view, x, device, adaptor=None):
    """Float16 outputs of the view's image head on float32 rows `x` (batches as the trainer's own saver), and, with
    an adaptor, the adaptor's output on the same float32 head output."""
    view.eval()
    out = np.empty((len(x), WIDTH), dtype=np.float16)
    ad = None if adaptor is None else np.empty((len(x), WIDTH), dtype=np.float16)
    batch = ms_train.CONFIG['batch_size']
    with torch.no_grad():
        for a in range(0, len(x), batch):
            xx = torch.from_numpy(np.ascontiguousarray(x[a:a + batch])).to(device)
            z = view['img'](xx)
            out[a:a + len(xx)] = z.cpu().numpy().astype(np.float16)
            if adaptor is not None:
                ad[a:a + len(xx)] = adaptor(z).cpu().numpy().astype(np.float16)
    return out, ad


class Exporter:
    """Stands in for `ms_train.save_output`: loads the aspect's best state and writes its files into `run_dir`."""

    def __init__(self, run_dir, genre, p, device):
        self.run_dir, self.genre, self.p, self.device = Path(run_dir), genre, p, device
        self.info, self.exports = {}, {}

    def __call__(self, s, view, inputs, state, path, config, start):
        aspect = config['aspect']
        view.load_state_dict(state['best_state'])           # the genre view's state holds the adaptor
        view.eval()
        adaptor = self.genre.adaptor if aspect == GENRE else None
        old, _ = infer_img(view, inputs['img'][s['subset_pid']], self.device)
        assert old.shape == (N_OLD, WIDTH)
        files = {f'{aspect}__old5000.npz': dict(img=old)}
        if self.p is not None:
            p_img, p_ad = infer_img(view, self.p['img'], self.device, adaptor)
            files[f'{aspect}__P.npz'] = dict(img=p_img)
            if p_ad is not None:
                files['genre__P_adaptor.npz'] = dict(adaptor=p_ad)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name, arrays in files.items():
            rb_part1.atomic_npz(self.run_dir / name, arrays)
        rb_part1.atomic_pt(self.run_dir / f'{aspect}.pt', state['best_state'])
        info = dict(aspect=aspect, best_epoch=state['best_epoch'], epochs_run=len(state['history']),
                    best_dev_loss=state['best'], history=state['history'], seconds=time.monotonic() - start,
                    arrays={n: {k: rb_part1.array_sha256(v) for k, v in a.items()} for n, a in files.items()})
        rb_part1.atomic_json(self.run_dir / f'{aspect}.json', info)
        self.info[aspect] = info
        self.exports[aspect] = {n: a for n, a in files.items() if n.endswith(('__P.npz', '__old5000.npz'))}
        return info


# ------------------------------------------------------------------ one run
def code_commit():
    git = ['git', '-C', str(eg_paths.ROUND)]
    head = subprocess.run(git + ['rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(git + ['status', '--porcelain', '-uno'], capture_output=True, text=True,
                                check=True).stdout.strip())
    return head, dirty


def write_run(run_dir, variant, seed, status, *, cause=None, failure_kind=None, exporter=None, p_sha256=None,
              smoke=False, max_epochs=None, seconds=None, old5000_problems=(), rerun_of=None):
    assert p_sha256, 'run.json needs p_sha256 (the P name list hash of the export; the scorer refuses a run without it)'
    commit, dirty = code_commit()
    info = exporter.info if exporter is not None else {}
    record = dict(variant=variant, seed=seed, switches=V.switches(variant), status=status,
                  best_epoch={a: i['best_epoch'] for a, i in info.items()},
                  epochs_run={a: i['epochs_run'] for a, i in info.items()},
                  best_dev_loss={a: i['best_dev_loss'] for a, i in info.items()},
                  p_sha256=p_sha256, smoke=smoke, max_epochs=max_epochs, seconds=seconds,
                  code_commit=commit, code_dirty=dirty, old5000_problems=list(old5000_problems))
    if rerun_of is not None:
        record['rerun_of'] = str(rerun_of)         # VG4: this run is the one rerun of a VG3 failure
    if status == 'failed':
        assert cause and failure_kind in ('nonfinite_loss', 'vg3')
        record.update(cause=cause, failure_kind=failure_kind)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    rb_part1.atomic_json(Path(run_dir) / 'run.json', record)
    return record


def train_cohort(s, inputs, graphs, manifests, recipe, variant, seed, device, run_dir, *, p, p_sha256,
                 smoke=False, max_epochs=None, rerun_of=None):
    """Train R6-5 with the genre change of `variant` at `seed`, export, check VG3, write `run.json`; returns it.

    `inputs` {aspect: {img, cap}} (the default images and caption means), `p` {img: float32 [N_P, 2048]} or None.
    A non-finite loss, or a P export failing VG3, gives a record with status failed and its cause. Problems of the old
    5,000 exports are listed in `old5000_problems` (rule VG3 covers P only). `rerun_of`: the moved-aside folder of the
    VG3-failed run this one repeats."""
    switches = V.switches(variant)
    run_dir, device = Path(run_dir), torch.device(device)
    max_epochs = ms_train.CONFIG['max_epochs'] if max_epochs is None else max_epochs
    genre = GenreLoss(switches, seed)
    if genre.w:
        fit, _ = ms_train.disjoint_split(s)
        teacher = ms_train.ags_trainimg.aspect_images(s, ms_train.head.ASPECTS[GENRE])
        genre.std = TorchStd(rb_probe.Standardizer(teacher[fit]), device)     # the 13,400 fit paintings' rows
        del teacher
    exporter = Exporter(run_dir, genre, p, device)
    start = time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(patch.object(ms_train, 'build_model', build_model_with(genre, ms_train.build_model)))
        stack.enter_context(patch.object(ms_train, 'head_loss', partial(
            checked_head_loss, switches=recipe['switches'], genre=genre)))
        stack.enter_context(patch.object(ms_train, 'save_output', exporter))
        try:
            ms_train.run(s, inputs, graphs, manifests, recipe['base'], variant, list(ASPECTS), seed, device,
                         run_dir / 'trainer', smoke, max_epochs)
        except NonFiniteLoss as error:
            return write_run(run_dir, variant, seed, 'failed', failure_kind='nonfinite_loss',
                             cause=f'non-finite loss in the {error.args[0]} head', p_sha256=p_sha256, smoke=smoke,
                             max_epochs=max_epochs, seconds=time.monotonic() - start, rerun_of=rerun_of)
    assert set(exporter.info) == set(ASPECTS)
    problems, old_problems = [], []
    for aspect in ASPECTS:
        for name, arrays in exporter.exports[aspect].items():
            old = name.endswith('__old5000.npz')
            problem = vg3_problem(arrays['img'], N_OLD if old else N_P)
            if problem:
                (old_problems if old else problems).append(f'{name}: {problem}')
    common = dict(exporter=exporter, p_sha256=p_sha256, smoke=smoke, max_epochs=max_epochs,
                  seconds=time.monotonic() - start, old5000_problems=old_problems, rerun_of=rerun_of)
    if problems:
        return write_run(run_dir, variant, seed, 'failed', failure_kind='vg3', cause='VG3: ' + '; '.join(problems),
                         **common)
    return write_run(run_dir, variant, seed, 'ok', **common)


# ------------------------------------------------------------------ data, runs, resume
class Data:
    """The frozen sample, its default inputs and P's default images, loaded once and only when a run needs them."""

    def __init__(self):
        self._loaded = None

    def get(self):
        if self._loaded is None:
            import rb_data
            s = ab_train.load_sample()
            inputs = ms_train.head.load_inputs(s, ASPECTS[0], 'default')
            p_sha = rb_data.p_sha256()
            assert p_sha == eg_paths.P_SHA256, f'P hash {p_sha}'
            img = rb_data.p_default_images()
            assert img.shape == (N_P, WIDTH) and img.dtype == np.float32 and np.isfinite(img).all()
            self._loaded = (s, inputs, dict(img=img), p_sha)
        return self._loaded


def resolved(seed):
    """R6-5's row with its graph names formatted for the seed."""
    recipe = rb_rows.recipe(BASE)
    recipe['base']['graphs'] = {a: g.format(seed=seed) for a, g in recipe['base']['graphs'].items()}
    return recipe


def check_outputs(outputs, smoke):
    outputs = Path(outputs).resolve()
    assert not smoke or eg_paths.smoke_ok(outputs), '--smoke outputs must be under /tmp/claude-0'
    return outputs


def move_aside(outputs, run_dir, variant, seed):
    """Keep an incomplete or failed run folder out of the way (deleting nothing): stage_iii/replaced/<variant>/."""
    target = Path(outputs) / 'stage_iii/replaced' / variant / f's{seed}-{time.time_ns()}'
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(run_dir).rename(target)
    return target


def vg3_failures_aside(outputs, variant, seed):
    """The moved-aside folders of this variant and seed whose run.json is a VG3 failure (an earlier rerun's origin)."""
    found = []
    for path in sorted((Path(outputs) / 'stage_iii/replaced' / variant).glob(f's{seed}-*/run.json')):
        record = json.loads(path.read_text())
        if record.get('status') == 'failed' and record.get('failure_kind') == 'vg3':
            found.append(path.parent)
    return found


def run_variant(data, variant, seed, outputs, device, *, run_dir=None, smoke=False, max_epochs=None,
                rerun_failed=False):
    """One run into `run_dir` (default `eg_paths.iii_run`); skipped when its run.json matches; returns the record.

    A folder without run.json (an interrupted run) is moved aside. A failed run stays as it is, unless `rerun_failed`
    and it failed VG3 (VG4 allows that one rerun, same seed and code); the old folder is moved aside."""
    V.parse(variant)
    assert seed in eg_paths.SEEDS
    outputs = check_outputs(outputs, smoke)
    run_dir = Path(run_dir) if run_dir is not None else eg_paths.iii_run(outputs, variant, seed)
    rerun_of = None
    max_epochs_used = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    assert not smoke or max_epochs_used == 2, '--smoke requires 2 epochs'
    record_path = run_dir / 'run.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        same = (record['variant'] == variant and record['seed'] == seed and record['switches'] == V.switches(variant)
                and record['smoke'] == smoke and record['max_epochs'] == max_epochs_used)
        assert same, f'{record_path}: a run of another configuration; move it aside'
        if record['status'] == 'ok' or not (rerun_failed and record['failure_kind'] == 'vg3'):
            print(f'{variant} s{seed}: {record["status"]} run exists; skip', flush=True)
            return record
        # VG4 allows one rerun, same seed and code: refuse a second one, and one on other code.
        assert 'rerun_of' not in record and not vg3_failures_aside(outputs, variant, seed), (
            f'{variant} s{seed}: already rerun once after a VG3 failure (VG4); no second rerun')
        head, dirty = code_commit()
        assert record['code_commit'] == head and not record['code_dirty'] and not dirty, (
            f'{variant} s{seed}: a rerun needs the same code: the failed run was {record["code_commit"]} '
            f'(dirty {record["code_dirty"]}), now {head} (dirty {dirty})')
        rerun_of = move_aside(outputs, run_dir, variant, seed)
        print(f'{variant} s{seed}: VG3 failure, rerun once; old folder moved to {rerun_of}', flush=True)
    elif run_dir.exists() and any(run_dir.iterdir()):
        print(f'{variant} s{seed}: incomplete folder moved to {move_aside(outputs, run_dir, variant, seed)}', flush=True)
    if torch.device(device).type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (queue)'
    eg_paths.check_imports()
    s, inputs, p, p_sha = data.get()
    recipe = resolved(seed)
    base = ab_train.require_graph_store(recipe)
    graphs, manifests = {}, {}
    for aspect in ASPECTS:
        graphs[aspect], manifests[aspect] = ms_train.head.load_graph(s, base, aspect, [recipe['base']['graphs'][aspect]])
    return train_cohort(s, {a: inputs for a in ASPECTS}, graphs, manifests, recipe, variant, seed, device, run_dir,
                        p=p, p_sha256=p_sha, smoke=smoke, max_epochs=max_epochs_used, rerun_of=rerun_of)


# ------------------------------------------------------------------ VG1 and the fresh-seed parity
def states_equal(a, b):
    """Two saved best states: same key order, every tensor torch.equal."""
    a, b = (torch.load(x, map_location='cpu', weights_only=True) for x in (a, b))
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def atomic_record(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rb_part1.atomic_json(Path(path), value)


def regress_vg1(data, outputs, device):
    """VG1, training half (rule §4.3): this trainer with every switch off trains R6-5 at seed 0 into
    stage_iii/regression/vg1_run; its P exports equal 12-24's bit for bit. Also compared, reported: the old-5,000
    exports and the best states."""
    outputs = check_outputs(outputs, False)
    run_dir = outputs / 'stage_iii/regression/vg1_run'
    run = run_variant(data, BASE, 0, outputs, device, run_dir=run_dir)
    ref = eg_paths.RB_PART1 / 'train' / BASE
    items = {}
    if run['status'] == 'ok':
        for aspect in ASPECTS:
            item = {}
            for label in ('P', 'old5000'):
                with np.load(run_dir / f'{aspect}__{label}.npz') as mine, \
                        np.load(ref / f'{aspect}__s0__{label}.npz') as old:
                    item[f'{label}_img_equal'] = bits_equal(mine['img'], old['img'])
            item['state_equal'] = states_equal(run_dir / f'{aspect}.pt', ref / f'{aspect}__s0.pt')
            items[aspect] = item
    # The rule's test is the P exports; the other two comparisons are reported.
    passed = run['status'] == 'ok' and all(i['P_img_equal'] for i in items.values()) and len(items) == len(ASPECTS)
    commit, dirty = code_commit()
    record = {'pass': passed, 'items': items, 'run_status': run['status'], 'run_cause': run.get('cause'),
              'reference': str(ref), 'run_dir': str(run_dir), 'code_commit': commit, 'code_dirty': dirty}
    atomic_record(outputs / 'stage_iii/vg1.json', record)
    return record


def parity_result(record):
    """Parity evidence without per-run commit, dirty state or timing bookkeeping."""
    if isinstance(record, dict):
        return {k: parity_result(v) for k, v in record.items() if k not in ('code_commit', 'code_dirty', 'seconds')}
    return record


def parity(data, outputs, device, seeds):
    """R6-5 at fresh seeds (into eg_paths.iii_run) against 12-24's K4 weights of the same seed (rule §4.3, C12)."""
    outputs = check_outputs(outputs, False)
    path = outputs / 'stage_iii/fresh_parity.json'
    record = json.loads(path.read_text()) if path.exists() else dict(seeds={})
    for seed in seeds:
        assert seed in V.FRESH_SEEDS, seed
        run = run_variant(data, BASE, seed, outputs, device)
        run_dir = eg_paths.iii_run(outputs, BASE, seed)
        commit, dirty = code_commit()
        item = dict(run_status=run['status'], run_cause=run.get('cause'), code_commit=commit, code_dirty=dirty)
        if run['status'] == 'ok':
            item['weights_equal'] = {a: states_equal(run_dir / f'{a}.pt', K4_WEIGHTS / f'{a}__s{seed}.pt')
                                     for a in ASPECTS}
            item['pass'] = all(item['weights_equal'].values())
        else:
            item['pass'] = False
        record['seeds'][str(seed)] = item
    commit, dirty = code_commit()                  # of the last call; every seed's own commit is in its item
    record.update(code_commit=commit, code_dirty=dirty, reference=str(K4_WEIGHTS))
    record['pass'] = all(record['seeds'].get(str(s), {}).get('pass') for s in V.FRESH_SEEDS)
    if path.exists():
        old = json.loads(path.read_text())
        if parity_result(old) == parity_result(record):
            return old
    atomic_record(path, record)
    return record


# ------------------------------------------------------------------ command line
@threadpool_limits.wrap(limits=8)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'regress', 'parity'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
        cmd.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        if name == 'train':
            cmd.add_argument('--variants', nargs='+', required=True, help="'R6-5' or a name of eg_variants (GT-2, ...)")
            cmd.add_argument('--seeds', type=int, nargs='+', default=[0])
            cmd.add_argument('--smoke', action='store_true', help='2 epochs, first 5000 fit pairs, outputs under /tmp/claude-0')
            cmd.add_argument('--max-epochs', type=int)
            cmd.add_argument('--rerun-failed', action='store_true', help='rerun a run that failed VG3, once (VG4)')
        elif name == 'regress':
            cmd.add_argument('gate', choices=['vg1'])
        else:
            cmd.add_argument('--seeds', type=int, nargs='+', default=list(V.FRESH_SEEDS))
    args = parser.parse_args(argv)
    t0 = time.monotonic()
    torch.set_num_threads(8)
    data = Data()
    if args.command == 'train':
        for variant in args.variants:
            V.parse(variant)
        for variant in args.variants:
            for seed in args.seeds:
                record = run_variant(data, variant, seed, args.outputs, args.device, smoke=args.smoke,
                                     max_epochs=args.max_epochs, rerun_failed=args.rerun_failed)
                print(f'EG_TRAIN3 {variant} s{seed} {record["status"]} {record.get("cause", "")}'.rstrip(), flush=True)
    elif args.command == 'regress':
        record = regress_vg1(data, args.outputs, args.device)
        print(f'EG_TRAIN3 VG1 pass={record["pass"]}', flush=True)
    else:
        record = parity(data, args.outputs, args.device, args.seeds)
        print(f'EG_TRAIN3 PARITY pass={record["pass"]} seeds={sorted(record["seeds"])}', flush=True)
    print(f'EG_TRAIN3 DONE seconds={time.monotonic() - t0:.0f}', flush=True)


if __name__ == '__main__':
    main()
