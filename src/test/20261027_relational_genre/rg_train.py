"""Sample-scale relational genre trainer: train, regress va1, eq and lossscale."""
import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import rg_paths as paths
import rg_variants as V
import rg_loss
import eg_train3 as eg
import ab_train
import rb_part1
from ab_paths import ms_train

BASE, ASPECTS = V.BASE, tuple(paths.ASPECTS)
NonFiniteLoss = eg.NonFiniteLoss
resolved = eg.resolved
bits_equal = eg.bits_equal
states_equal = eg.states_equal
code_commit = eg.code_commit
atomic_record = eg.atomic_record


def unit_rows(x):
    assert x.ndim == 2 and x.dtype == torch.float32 and not x.requires_grad
    assert bool(torch.isfinite(x).all())
    assert bool((torch.linalg.vector_norm(x, dim=1) - 1).abs().lt(1e-3).all()), 'teacher rows must be unit norm'
    return x


def build_model_with(genre, original, fit_teacher, *, diagnostic=False):
    """Sheet factory, then fit-local anchors from a private generator; no model/RNG changes."""
    def build(row, aspects, device):
        models, views = original(row, aspects, device)
        if genre.n_anchors or diagnostic:
            assert 'genre' in views and genre.anchors is None
            teacher = unit_rows(fit_teacher())
            ids = rg_loss.draw_anchors(len(teacher), genre.seed)
            genre.n_fit = len(teacher)
            genre.anchor_ids = ids.to(device)
            genre.anchors = teacher[ids.to(teacher.device)].detach().to(device)
        return models, views
    return build


def fit_teacher_rows(s, graph, device):
    """Use exactly the node order used by ms_train.run to construct its fit tensor."""
    teacher = ms_train.ags_trainimg.aspect_images(s, ms_train.head.ASPECTS['genre'])
    assert teacher.ndim == 2 and teacher.dtype == np.float32 and np.isfinite(teacher).all()
    ids = graph['fit_node_ids']
    assert ids.dtype == np.int64 and ids.ndim == 1 and len(np.unique(ids)) == len(ids)
    assert (ids >= 0).all() and (ids < len(teacher)).all()
    return unit_rows(torch.from_numpy(np.ascontiguousarray(teacher[ids])).to(device))


class TrainingLoss:
    """Route every aspect through rg_loss; observe only genre training calls, never dev calls."""
    def __init__(self, genre, switches, n_pairs, *, diagnostic=False):
        assert n_pairs > 0
        self.genre, self.switches, self.n_pairs = genre, switches, n_pairs
        self.diagnostic, self.steps = diagnostic, []
        self.epoch, self.seen = 1, 0

    def __call__(self, models, img, cap, pairs, weights, row, aspect, teacher=None, extras=None,
                 include_mask_penalty=True):
        training = aspect == 'genre' and include_mask_penalty and torch.is_grad_enabled()
        outputs = []
        handle = None
        if training and self.diagnostic:
            handle = models['img'].register_forward_hook(lambda _m, _a, z: outputs.append(z.detach()))
        try:
            loss = rg_loss.switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher, extras,
                                              include_mask_penalty, switches=self.switches, genre=self.genre)
        finally:
            if handle is not None:
                handle.remove()
        if not bool(torch.isfinite(loss)):
            raise NonFiniteLoss(aspect)
        if training:
            terms = {k: float(v) for k, v in self.genre.term_log.items()}
            if self.diagnostic:
                assert len(outputs) == 1 and self.genre.anchors is not None
                ids = torch.cat(tuple(torch.as_tensor(pairs, device=img.device).unbind(1)))
                with torch.no_grad():
                    terms['anchors'] = float(rg_loss.anchors_loss(
                        outputs[0], teacher[ids], ids, self.genre.anchors, self.genre.anchors,
                        self.genre.anchor_ids))
            if not all(np.isfinite(v) for v in terms.values()):
                raise NonFiniteLoss(aspect)
            self.steps.append(dict(step=len(self.steps) + 1, epoch=self.epoch, pairs=len(pairs), **terms))
            self.seen += len(pairs)
            assert self.seen <= self.n_pairs
            if self.seen == self.n_pairs:
                self.epoch, self.seen = self.epoch + 1, 0
        return loss

    def means(self):
        result = []
        for epoch in sorted({s['epoch'] for s in self.steps}):
            steps = [s for s in self.steps if s['epoch'] == epoch]
            keys = set(steps[0]) - {'step', 'epoch', 'pairs'}
            assert all(set(s) - {'step', 'epoch', 'pairs'} == keys for s in steps)
            result.append(dict(epoch=epoch, calls=len(steps), pairs=sum(s['pairs'] for s in steps),
                               **{k: float(np.mean([s[k] for s in steps])) for k in sorted(keys)}))
        return result


def anchor_record(genre):
    if genre.anchors is None:
        return None
    return dict(count=len(genre.anchors), positions_sha256=rb_part1.array_sha256(
        genre.anchor_ids.detach().cpu().numpy().astype(np.int64)), n_fit=genre.n_fit)


class Exporter:
    """Copied from eg_train3.Exporter, dropping GA and adding provenance to each JSON."""
    def __init__(self, run_dir, p, device, input_hashes):
        self.run_dir, self.p, self.device = Path(run_dir), p, device
        self.input_hashes = input_hashes
        self.info, self.exports = {}, {}

    def __call__(self, s, view, inputs, state, path, config, start):
        aspect = config['aspect']
        view.load_state_dict(state['best_state'])
        view.eval()
        old, _ = eg.infer_img(view, inputs['img'][s['subset_pid']], self.device)
        assert old.shape == (paths.N_OLD, paths.WIDTH)
        files = {f'{aspect}__old5000.npz': dict(img=old)}
        if self.p is not None:
            p_img, _ = eg.infer_img(view, self.p['img'], self.device)
            files[f'{aspect}__P.npz'] = dict(img=p_img)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name, arrays in files.items():
            rb_part1.atomic_npz(self.run_dir / name, arrays)
        rb_part1.atomic_pt(self.run_dir / f'{aspect}.pt', state['best_state'])
        commit, dirty = code_commit()
        info = dict(aspect=aspect, best_epoch=state['best_epoch'], epochs_run=len(state['history']),
                    best_dev_loss=state['best'], history=state['history'], seconds=time.monotonic() - start,
                    arrays={n: {k: rb_part1.array_sha256(v) for k, v in a.items()} for n, a in files.items()},
                    code_commit=commit, code_dirty=dirty, input_hashes=self.input_hashes, config=config)
        atomic_record(self.run_dir / f'{aspect}.json', info)
        self.info[aspect], self.exports[aspect] = info, files
        return info


def write_run(run_dir, arm, seed, status, *, genre, loss, exporter, p_sha256, smoke, max_epochs,
              seconds, input_hashes, cause=None, failure_kind=None, rerun_of=None):
    assert p_sha256
    commit, dirty = code_commit()
    info = exporter.info
    record = dict(variant=arm, seed=seed, switches=genre.switches, status=status,
                  best_epoch={a: i['best_epoch'] for a, i in info.items()},
                  epochs_run={a: i['epochs_run'] for a, i in info.items()},
                  best_dev_loss={a: i['best_dev_loss'] for a, i in info.items()},
                  p_sha256=p_sha256, smoke=smoke, max_epochs=max_epochs, seconds=seconds,
                  term_means=loss.means(), term_aspect='genre', term_reduction='mean_over_training_calls',
                  code_commit=commit, code_dirty=dirty, input_hashes=input_hashes,
                  old5000_problems=[])
    anchors = anchor_record(genre)
    if anchors is not None:
        record['anchors'] = anchors
    if rerun_of is not None:
        record['rerun_of'] = str(rerun_of)
    if status == 'failed':
        assert cause and failure_kind in ('nonfinite_loss', 'va3')
        record.update(cause=cause, failure_kind=failure_kind)
    for aspect, files in exporter.exports.items():
        for name, arrays in files.items():
            if name.endswith('__old5000.npz'):
                problem = eg.vg3_problem(arrays['img'], paths.N_OLD)
                if problem:
                    record['old5000_problems'].append(f'{name}: {problem}')
    atomic_record(Path(run_dir) / 'run.json', record)
    return record


def train_cohort(s, inputs, graphs, manifests, recipe, variant, seed, device, run_dir, *, p, p_sha256,
                 smoke=False, max_epochs=None, rerun_of=None, diagnostic=False, input_hashes=None):
    """Copied from eg_train3.train_cohort: unchanged sheet loop, rg loss/build/save hooks, VA3."""
    sw = dict(V.LOSS_SCALE_SWITCHES) if diagnostic else V.switches(variant)
    assert not diagnostic or (variant, seed, max_epochs, p) == (V.LOSS_SCALE, 0, 1, None)
    assert p is None or p_sha256 == paths.P_SHA256, 'VA3 P name hash mismatch'
    run_dir, device = Path(run_dir), torch.device(device)
    max_epochs = ms_train.CONFIG['max_epochs'] if max_epochs is None else max_epochs
    genre = rg_loss.GenreLoss(sw, seed, term_log={})
    n_pairs = len(graphs['genre']['rows'][recipe['base']['graphs']['genre']]['fit'][0])
    loss = TrainingLoss(genre, recipe['switches'], min(n_pairs, 5000) if smoke else n_pairs,
                        diagnostic=diagnostic)
    input_hashes = {} if input_hashes is None else input_hashes
    exporter = Exporter(run_dir, p, device, input_hashes)
    start = time.monotonic()
    status, failure_kind, cause = 'ok', None, None
    with ExitStack() as stack:
        stack.enter_context(patch.object(ms_train, 'build_model', build_model_with(
            genre, ms_train.build_model, lambda: fit_teacher_rows(s, graphs['genre'], device),
            diagnostic=diagnostic)))
        stack.enter_context(patch.object(ms_train, 'head_loss', loss))
        stack.enter_context(patch.object(ms_train, 'save_output', exporter))
        try:
            ms_train.run(s, inputs, graphs, manifests, recipe['base'], variant, list(ASPECTS), seed, device,
                         run_dir / 'trainer', smoke, max_epochs)
        except NonFiniteLoss as error:
            status, failure_kind, cause = 'failed', 'nonfinite_loss', f'non-finite loss in the {error.args[0]} head'
    if status == 'ok':
        assert set(exporter.info) == set(ASPECTS)
        problems = []
        if p is not None:
            for aspect in ASPECTS:
                x = exporter.exports[aspect][f'{aspect}__P.npz']['img']
                problem = eg.vg3_problem(x, paths.N_P)
                if problem:
                    problems.append(f'{aspect}__P.npz: {problem}')
        if problems:
            status, failure_kind, cause = 'failed', 'va3', 'VA3: ' + '; '.join(problems)
    if diagnostic:
        commit, dirty = code_commit()
        atomic_record(run_dir.parent / 'loss_scale.json', dict(
            status=status, cause=cause, variant=variant, seed=seed, switches=sw, smoke=smoke, max_epochs=1,
            steps=loss.steps, epoch_means=loss.means(), reduction='mean_over_training_calls',
            anchors=anchor_record(genre), code_commit=commit, code_dirty=dirty, input_hashes=input_hashes))
    record = write_run(run_dir, variant, seed, status, genre=genre, loss=loss, exporter=exporter,
                       p_sha256=p_sha256, smoke=smoke, max_epochs=max_epochs, seconds=time.monotonic() - start,
                       input_hashes=input_hashes, cause=cause, failure_kind=failure_kind, rerun_of=rerun_of)
    return record


class Data:
    """Frozen sample and default inputs; P images are read only when exporting on P."""
    def __init__(self):
        self.sample = self.inputs = self.p = self.hashes = None

    def get(self, *, export_p=True):
        import rb_data
        if self.sample is None:
            self.sample = ab_train.load_sample()
            self.inputs = ms_train.head.load_inputs(self.sample, ASPECTS[0], 'default')
            assert set(self.inputs) == {'img', 'cap'}
            for x in self.inputs.values():
                assert x.shape == (19889, paths.WIDTH) and x.dtype == np.float32 and np.isfinite(x).all()
            self.hashes = dict(sample_sha256=ms_train.head.fingerprint(self.sample),
                               default_inputs={k: rb_part1.array_sha256(x) for k, x in self.inputs.items()},
                               teachers={})
            for aspect in ASPECTS:
                teacher = ms_train.ags_trainimg.aspect_images(self.sample, ms_train.head.ASPECTS[aspect])
                assert teacher.shape == (19889, paths.WIDTH) and teacher.dtype == np.float32
                unit_rows(torch.from_numpy(teacher))
                self.hashes['teachers'][aspect] = rb_part1.array_sha256(teacher)
        p_sha = rb_data.p_sha256()
        assert p_sha == paths.P_SHA256, f'P hash {p_sha}'
        if export_p and self.p is None:
            img = rb_data.p_default_images()
            assert img.shape == (paths.N_P, paths.WIDTH) and img.dtype == np.float32 and np.isfinite(img).all()
            self.p = dict(img=img)
            self.hashes['p_default_img'] = rb_part1.array_sha256(img)
        return self.sample, self.inputs, self.p if export_p else None, p_sha


def load_graphs(s, recipe):
    base = ab_train.require_graph_store(recipe)
    graphs, manifests = {}, {}
    for aspect in ASPECTS:
        graphs[aspect], manifests[aspect] = ms_train.head.load_graph(s, base, aspect, [recipe['base']['graphs'][aspect]])
    return graphs, manifests


def input_identity(data, p_sha, manifests):
    return dict(data.hashes, p_sha256=p_sha,
                graph_archives={a: m['archives'][a] for a, m in manifests.items()},
                graph_identity={a: m['identity'] for a, m in manifests.items()})


def check_outputs(outputs, smoke):
    outputs = Path(outputs).resolve()
    assert not smoke or paths.smoke_ok(outputs), '--smoke outputs must be under /tmp/claude-0'
    for frozen in (paths.EG_OUT, paths.RB_OUT, paths.CT_OUT, paths.PRE):
        assert not outputs.is_relative_to(frozen.resolve()), f'outputs inside frozen input {frozen}'
    return outputs


def check_device(device):
    device = torch.device(device)
    if device.type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (queue)'
    paths.check_imports()
    return device


def move_aside(outputs, run_dir, arm, seed):
    # Copied from eg_train3.move_aside: preserve interrupted/failed artifacts, never delete.
    target = Path(outputs) / 'replaced' / arm / f's{seed}-{time.time_ns()}'
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(run_dir).rename(target)
    return target


def va3_failures_aside(outputs, arm, seed):
    found = []
    for path in sorted((Path(outputs) / 'replaced' / arm).glob(f's{seed}-*/run.json')):
        record = json.loads(path.read_text())
        assert record['variant'] == arm and record['seed'] == seed
        if record['status'] == 'failed' and record['failure_kind'] == 'va3':
            found.append(path.parent)
    return found


def run_variant(data, variant, seed, outputs, device, *, run_dir=None, smoke=False, max_epochs=None,
                diagnostic=False):
    """Copied from eg_train3.run_variant; rg arms/paths and an automatic single VA3 rerun."""
    switches = dict(V.LOSS_SCALE_SWITCHES) if diagnostic else V.switches(variant)
    assert seed in (0, 1, 2, 3)
    assert not diagnostic or (variant, seed) == (V.LOSS_SCALE, 0)
    outputs, device = check_outputs(outputs, smoke), check_device(device)
    run_dir = Path(run_dir).resolve() if run_dir is not None else paths.run_dir(outputs, variant, seed)
    assert run_dir.is_relative_to(outputs), 'run folder must be under --outputs'
    max_epochs = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    assert not smoke or diagnostic or max_epochs == 2, '--smoke requires 2 epochs'
    assert 1 <= max_epochs <= ms_train.CONFIG['max_epochs']
    s, inputs, p, p_sha = data.get(export_p=not diagnostic)
    recipe = resolved(seed)
    graphs, manifests = load_graphs(s, recipe)
    identity = input_identity(data, p_sha, manifests)
    rerun_of = None
    record_path = run_dir / 'run.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert (record['variant'], record['seed'], record['switches'], record['smoke'], record['max_epochs']) == (
            variant, seed, switches, smoke, max_epochs), f'{record_path}: configuration mismatch'
        assert record['p_sha256'] == p_sha and record['input_hashes'] == identity, 'run inputs changed'
        assert record['status'] in ('ok', 'failed')
        if record['status'] == 'ok' or record['failure_kind'] != 'va3' or 'rerun_of' in record:
            if record['status'] == 'ok':
                verify_cached_run(run_dir, record, export_p=not diagnostic)
            print(f'{variant} s{seed}: {record["status"]} run exists; skip', flush=True)
            return record
        assert not va3_failures_aside(outputs, variant, seed), 'already rerun once after VA3'
        head, dirty = code_commit()
        assert record['code_commit'] == head and not record['code_dirty'] and not dirty, 'VA3 rerun needs same clean code'
        rerun_of = move_aside(outputs, run_dir, variant, seed)
    else:
        # An interrupted rerun must retain its origin, even if its folder is absent/empty.
        previous = va3_failures_aside(outputs, variant, seed)
        assert len(previous) <= 1, 'already rerun once after VA3'
        if previous:
            origin = json.loads((previous[0] / 'run.json').read_text())
            assert (origin['code_commit'], origin['code_dirty']) == code_commit() and not origin['code_dirty']
            assert (origin['variant'], origin['seed'], origin['switches'], origin['smoke'], origin['max_epochs']) == (
                variant, seed, switches, smoke, max_epochs), 'interrupted VA3 rerun configuration changed'
            assert origin['input_hashes'] == identity, 'interrupted VA3 rerun inputs changed'
            rerun_of = previous[0]
        if run_dir.exists() and any(run_dir.iterdir()):
            move_aside(outputs, run_dir, variant, seed)
    record = train_cohort(s, {a: inputs for a in ASPECTS}, graphs, manifests, recipe, variant, seed, device, run_dir,
                          p=p, p_sha256=p_sha, smoke=smoke, max_epochs=max_epochs, rerun_of=rerun_of,
                          diagnostic=diagnostic, input_hashes=identity)
    if record['status'] == 'failed' and record['failure_kind'] == 'va3' and rerun_of is None:
        # The same resume checks also apply to the immediate, rule-mandated rerun.
        return run_variant(data, variant, seed, outputs, device, run_dir=run_dir, smoke=smoke,
                           max_epochs=max_epochs, diagnostic=diagnostic)
    return record


def read_export(path, n_rows):
    with np.load(path, allow_pickle=False) as z:
        assert 'img' in z.files
        x = z['img']
    assert eg.vg3_problem(x, n_rows) is None, f'{path}: invalid export'
    return x


def verify_cached_run(folder, record, *, export_p):
    """An existing successful run must still have its recorded exports and best states."""
    assert set(record['best_epoch']) == set(ASPECTS)
    for aspect in ASPECTS:
        meta = json.loads((folder / f'{aspect}.json').read_text())
        assert meta['aspect'] == aspect and meta['best_epoch'] == record['best_epoch'][aspect]
        assert meta['input_hashes'] == record['input_hashes']
        assert (folder / f'{aspect}.pt').is_file()
        for label, n in (('old5000', paths.N_OLD), ('P', paths.N_P)):
            if label == 'P' and not export_p:
                continue
            name = f'{aspect}__{label}.npz'
            with np.load(folder / name, allow_pickle=False) as z:
                assert set(z.files) == {'img'}
                x = z['img']
            assert x.shape == (n, paths.WIDTH) and x.dtype == np.float16
            assert rb_part1.array_sha256(x) == meta['arrays'][name]['img'], 'cached export hash mismatch'
            if label == 'P':
                assert eg.vg3_problem(x, n) is None, 'cached export fails VA3'


def regress_va1(data, outputs, device, *, smoke=False):
    """VA1 training half: the three prescribed replays, P array bits deciding; states reported."""
    outputs = check_outputs(outputs, smoke)
    items, hashes = {}, {}
    for arm, seed in ((BASE, 0), ('GR-8', 0), (BASE, 1)):
        folder = outputs / 'regression' / f'va1_{arm}_s{seed}'
        run = run_variant(data, arm, seed, outputs, device, run_dir=folder, smoke=smoke)
        base = arm == BASE and seed == 0
        ref = paths.RB_PART1 / 'train' / BASE if base else paths.frozen_run(arm, seed)
        if not base:
            frozen = json.loads((ref / 'run.json').read_text())
            assert frozen['status'] == 'ok' and frozen['variant'] == arm and frozen['seed'] == seed
            assert frozen['p_sha256'] == paths.P_SHA256
        aspects = {}
        if run['status'] == 'ok':
            for aspect in ASPECTS:
                stem = f'{aspect}__s0' if base else aspect
                meta = json.loads((ref / (f'{stem}.json' if base else 'run.json')).read_text())
                assert meta['p_sha256'] == paths.P_SHA256
                comparisons = {}
                for label, n in (('P', paths.N_P), ('old5000', paths.N_OLD)):
                    mine = read_export(folder / f'{aspect}__{label}.npz', n)
                    old = read_export(ref / f'{stem}__{label}.npz', n)
                    comparisons[f'{label}_img_equal'] = bits_equal(mine, old)
                    hashes[f'{arm}_s{seed}/{aspect}/{label}'] = dict(
                        replay=rb_part1.array_sha256(mine), reference=rb_part1.array_sha256(old))
                mine_state, old_state = folder / f'{aspect}.pt', ref / f'{stem}.pt'
                comparisons['state_equal'] = states_equal(mine_state, old_state)
                hashes[f'{arm}_s{seed}/{aspect}/state'] = dict(
                    replay=ms_train.head.sha_file(mine_state), reference=ms_train.head.sha_file(old_state))
                aspects[aspect] = comparisons
        passed = run['status'] == 'ok' and len(aspects) == len(ASPECTS) and all(
            a['P_img_equal'] for a in aspects.values())
        items[f'{arm}_s{seed}'] = dict(pass_=passed, aspects=aspects, run_status=run['status'],
                                       run_cause=run.get('cause'), reference=str(ref), run_dir=str(folder))
        items[f'{arm}_s{seed}']['pass'] = items[f'{arm}_s{seed}'].pop('pass_')
    commit, dirty = code_commit()
    record = dict(pass_=all(i['pass'] for i in items.values()) and not smoke,
                  items=items, smoke=smoke, p_sha256=paths.P_SHA256,
                  code_commit=commit, code_dirty=dirty, input_hashes=hashes)
    record['pass'] = record.pop('pass_')
    atomic_record(outputs / 'regression/va1.json', record)
    return record


def equivalence(data, outputs, device='cpu', *, smoke=False):
    """EQ on the trainer's first genre batch at seeds 0, 1, 2; no training or P exports."""
    outputs, device = check_outputs(outputs, smoke), check_device(device)
    s, inputs, _, p_sha = data.get(export_p=False)
    prepared, hashes = [], {}
    for seed in (0, 1, 2):
        recipe = resolved(seed)
        graphs, manifests = load_graphs(s, recipe)
        graph = graphs['genre']
        pairs = graph['rows'][recipe['base']['graphs']['genre']]['fit'][0]
        if smoke:
            pairs = pairs[:5000]
        order = np.random.default_rng([seed, 2]).permutation(len(pairs))
        batch = pairs[order[:ms_train.CONFIG['batch_size']]]
        prepared.append((seed, recipe, graph, batch))
        hashes[str(seed)] = dict(input_identity(data, p_sha, manifests),
                                 batch_pairs=rb_part1.array_sha256(batch))
    path = outputs / 'regression/eq.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['smoke'] == smoke and old['device'] == str(device), 'EQ configuration mismatch'
        assert old['input_hashes'] == hashes and old['tolerance'] == V.TOL, 'EQ inputs/configuration changed'
        assert [i['seed'] for i in old['items']] == [0, 1, 2] and 'pass' in old
        return old
    items = []
    for seed, recipe, graph, batch in prepared:
        ids = torch.cat(tuple(torch.as_tensor(batch, device=device).unbind(1)))
        np.random.seed(seed)
        torch.manual_seed(seed)
        models, views = ms_train.build_model(recipe['base'], list(ASPECTS), device)
        models.eval()
        img = torch.from_numpy(np.ascontiguousarray(inputs['img'][graph['fit_node_ids']])).to(device)
        teacher = fit_teacher_rows(s, graph, device)
        assert img.shape == teacher.shape
        with torch.no_grad():
            z = views['genre']['img'](img[ids])
            t = teacher[ids]
            relational = float(ms_train.distill_loss(z, t, ids, 'relational'))
            anchors = float(rg_loss.anchors_loss(z, t, ids, z, t, ids))
        finite = bool(np.isfinite([relational, anchors]).all())
        # JSON never contains NaN/Inf, including failed gates.
        items.append(dict(seed=seed, rows=len(ids), unique_paintings=len(ids.unique()),
                          pairs=len(batch), relational=relational if finite else None,
                          anchors=anchors if finite else None,
                          difference=abs(relational - anchors) if finite else None))
        items[-1]['pass'] = finite and abs(relational - anchors) <= V.TOL
        del models, views, img, teacher, z, t
    commit, dirty = code_commit()
    record = dict(tolerance=V.TOL, items=items, smoke=smoke, device=str(device),
                  code_commit=commit, code_dirty=dirty, input_hashes=hashes)
    record['pass'] = all(i['pass'] for i in items)
    atomic_record(path, record)
    return record


@threadpool_limits.wrap(limits=8)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'regress', 'eq', 'lossscale'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--outputs', type=Path, default=paths.OUTPUTS)
        cmd.add_argument('--device', choices=['cuda', 'cpu'], default='cpu' if name == 'eq' else 'cuda')
        cmd.add_argument('--smoke', action='store_true', help='scratch outputs only; 5000 fit pairs, 2 epochs (loss scale: 1)')
        if name == 'train':
            cmd.add_argument('--arms', nargs='+', choices=list(V.ARMS), required=True)
            cmd.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2, 3], default=[0])
        elif name == 'regress':
            cmd.add_argument('gate', choices=['va1'])
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    outputs = check_outputs(args.outputs, args.smoke)
    data = Data()
    if args.command == 'train':
        failed = False
        for arm in args.arms:
            for seed in args.seeds:
                record = run_variant(data, arm, seed, outputs, args.device, smoke=args.smoke)
                print(f'RG_TRAIN {arm} s{seed} {record["status"]} {record.get("cause", "")}'.rstrip(), flush=True)
                failed |= record['status'] != 'ok'
        return int(failed)
    elif args.command == 'regress':
        record = regress_va1(data, outputs, args.device, smoke=args.smoke)
        print(f'RG_TRAIN VA1 pass={record["pass"]}', flush=True)
        return 0 if record['pass'] else 1
    elif args.command == 'eq':
        record = equivalence(data, outputs, args.device, smoke=args.smoke)
        print(f'RG_TRAIN EQ pass={record["pass"]}', flush=True)
        return 0 if record['pass'] else 1
    else:
        record = run_variant(data, V.LOSS_SCALE, 0, outputs, args.device, run_dir=outputs / 'loss_scale/run',
                             smoke=args.smoke, max_epochs=1, diagnostic=True)
        print(f'RG_TRAIN LOSSSCALE {record["status"]}', flush=True)
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
