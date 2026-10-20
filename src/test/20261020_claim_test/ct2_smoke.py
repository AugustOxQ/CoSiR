"""End-to-end smoke of the stage 2 runner on small synthetic stores, into a scratch root.

Every ct2_run step runs through ct2_run.main on seeds 0 to 4 and one epoch: the real graph build, D2 and
PercepT trainers and held writers, R-COL, scorer, bootstrap (8 resamples) and diagnostics code, on synthetic
features (1,500 fit and 150 validation paintings with two captions each, 12,281 held paintings). The
stage 2a gate is replaced by a recorder (stage 2a itself needs the real reference artifacts and the GPU):
its step runs the real ct2_stage2a.main with train and check replaced. The extract step runs with fake
stores and extractor. Nothing is written outside SCRATCH; production folders and reports are compared
before and after. This is not a stage 2a pass and licenses no claim result.

  ct2_smoke.py SCRATCH --device {cuda,cpu} [--steps extract,graphs,...]

--steps (default: all) allows a rerun of later steps in an existing SCRATCH, as the CPU smoke may not fit in
one command. On cpu the PercepT steps run with --smoke (the trainers refuse a non-smoke run off the GPU).
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
MAIN_ROUND = Path('/project/CoSiR/src/test/20261020_claim_test')
N_FIT, N_DEV, N_HELD, CAPTIONS = 1500, 150, 12281, 2
STEPS = ('budget', 'extract', 'graphs', 'stage2a', 'd2', 'core', 'paper', 'rcol', 'held', 'verdict',
         'diagnostics', 'gates')
SEEDS = tuple(range(5))   # ct_rcol.write_stage1 reads seeds 0 to 4 unconditionally


def scratch_root(path, resume):
    """Never let a smoke write to production artifacts or a shared feature store."""
    path = Path(path).resolve()
    forbidden = [HERE, MAIN_ROUND, Path('/data'), Path('/project/CoSiR')]
    assert not any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in forbidden), \
        'smoke needs a separate scratch output root'
    assert resume or not path.exists(), 'smoke output root already exists; use a new scratch directory'
    return path


def production_state():
    """Names, sizes and times of every production file a stage 2 step could write."""
    state = {}
    for base in (HERE / 'outputs', MAIN_ROUND / 'outputs/stage2'):
        for path in sorted(base.rglob('*')) if base.exists() else []:
            if 'logs' in path.relative_to(base).parts or path.name.startswith('queue_'):
                continue   # other sessions' queues write these
            stat = path.stat()
            state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    for name in ('stage2a.json', 'verdict_stage2.json', 'diag_stage2.json'):
        for folder in (HERE, MAIN_ROUND):
            path = folder / name
            state[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None
    return state


def l2(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class World:
    """Deterministic synthetic features with cluster structure, joined by painting name."""

    def __init__(self):
        rng = np.random.default_rng(91)
        self.fit_names = [f'syn-fit-{i:05d}' for i in range(N_FIT)]
        self.dev_names = [f'syn-val-{i:05d}' for i in range(N_DEV)]
        self.held_names = [f'syn-held-{i:05d}' for i in range(N_HELD)]
        self.names = np.asarray(self.fit_names + self.dev_names)
        self.all = self.fit_names + self.dev_names + self.held_names
        self.lookup = {name: i for i, name in enumerate(self.all)}
        total, k = len(self.all), 12
        self.cluster = rng.integers(k, size=total)

        def features(width):
            centers = rng.normal(size=(k, width))
            return l2(centers[self.cluster] * .7 + rng.normal(size=(total, width)))

        self.qwen, self.clip = features(2048), features(512)
        n = len(self.names)
        self.pid = np.repeat(np.arange(n, dtype=np.int64), CAPTIONS)
        noise = rng.normal(size=(n * CAPTIONS, 2048)).astype(np.float32) * .3
        self.cap = l2(self.qwen[self.pid] + noise)
        self.cap_emotion = l2(np.roll(self.qwen[self.pid], 5, axis=1) + noise)
        self.clip_text = l2(self.clip[self.pid] + rng.normal(size=(n * CAPTIONS, 512)).astype(np.float32) * .3)
        centers = rng.normal(size=(k, 768))
        self.roberta = (centers[self.cluster[self.pid]] + rng.normal(size=(n * CAPTIONS, 768))).astype(np.float32)
        self.labels = dict(emotion=self.cluster % 9, style=self.cluster % 8, genre=self.cluster % 4)

    def rows(self, names):
        return np.asarray([self.lookup[name] for name in names], dtype=np.int64)

    def paint_mean(self, per_caption):
        return l2(per_caption.reshape(len(self.names), CAPTIONS, -1).mean(axis=1))

    def patches(self, names):
        x = self.clip[self.rows(names)]
        noise = np.random.default_rng(3).normal(size=(50, 512)).astype(np.float32) * .05
        return (x[:, None, :] + noise[None]).astype(np.float32)

    def split(self, mode):
        assert mode == 'full'
        import ct2_data
        n = len(self.names)
        fit, dev = np.arange(N_FIT, dtype=np.int64), np.arange(N_FIT, n, dtype=np.int64)
        sample_ids = np.arange(n * CAPTIONS, dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        emotion = self.labels['emotion'][self.pid]
        s = dict(sample_ids=sample_ids, rows=sample_ids.copy(), pid=self.pid.copy(),
                 first=np.arange(n, dtype=np.int64) * CAPTIONS, all_painting_ids=self.names.copy(),
                 painting_ids=self.names[self.pid], subset_pid=empty, subset_image_indices=empty.copy(),
                 subset_painting_ids=np.empty(0, dtype=self.names.dtype), emo=emotion,
                 sty=self.labels['style'][self.pid], gen=self.labels['genre'][self.pid])
        s.update(mode=mode, fit=fit, dev=dev, fit_pairs=np.flatnonzero(np.isin(s['pid'], fit)),
                 names_sha256={role: ct2_data._names_sha256(self.names[rows])
                               for role, rows in (('fit', fit), ('dev', dev), ('eval', empty))})
        return s

    def default_images(self, s):
        return self.qwen[:len(self.names)].copy()

    def default_captions(self, s, *, per_caption=False):
        return self.cap.copy() if per_caption else self.paint_mean(self.cap)

    def instructed_images(self, s, key):
        shift = dict(emotion_focus=1, style_word=2, genre_word=3)[key]
        return np.roll(self.qwen[:len(self.names)], shift, axis=1).copy()

    def instructed_captions(self, s, key, *, per_caption=False):
        assert key == 'emotion_focus'
        return self.cap_emotion.copy() if per_caption else self.paint_mean(self.cap_emotion)

    def caption_features(self, s, kind):
        if kind == 'clip_text':
            return self.clip_text.copy()
        if kind == 'roberta':
            return self.roberta.copy()
        if kind == 'qwen_default':
            return self.default_captions(s, per_caption=True)
        assert kind == 'qwen_emotion_focus', kind
        return self.instructed_captions(s, 'emotion_focus', per_caption=True)

    def image_features(self, s, kind):
        if kind == 'qwen_default':
            return self.default_images(s)
        assert kind == 'clip_image', kind
        return self.clip[:len(self.names)].copy()

    def painting_labels(self, names):
        rows = self.rows(names)
        return {aspect: y[rows].astype(np.int64) for aspect, y in self.labels.items()}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(root, device='cpu', steps=STEPS):
    """Run the chosen steps; return the smoke report. Every assertion is on SCRATCH files."""
    started = time.perf_counter()
    import ct_held
    import ct_diag
    import ct_paper_sept as paper
    import ct_score as sc
    import ct2_d2
    import ms_train
    import ct2_data
    import ct2_diag
    import ct2_extract
    import ct2_percept as pt
    import ct2_run
    import ct2_stage2a
    import ct2_verdict

    steps = tuple(steps)
    assert set(steps) <= set(STEPS), steps
    root = scratch_root(root, resume=steps != STEPS)
    root.mkdir(parents=True, exist_ok=True)
    before = production_state()
    outputs, reports, full = root / 'outputs/stage2', root / 'reports', root / 'outputs/stage2/full'
    world = World()
    frozen_list = root / 'stage1_held/held_paintings.json'
    if not frozen_list.exists():
        frozen_list.parent.mkdir(parents=True, exist_ok=True)
        frozen_list.write_text(json.dumps(dict(names=world.held_names,
                                               sha256=ct_held.names_sha256(world.held_names))))
    smoke = device == 'cpu'
    report = dict(smoke=True, device=device, seeds=list(SEEDS), epochs=1, steps=[])
    gate_calls = []

    def passed(name):
        report['steps'].append(name)
        save_json(root / 'smoke.json', report)
        print(f'SMOKE {name} PASS', flush=True)

    def call(*argv):
        return ct2_run.main([*argv, '--root', str(outputs), '--reports', str(reports), '--device', device]
                            + (['--smoke'] if smoke else []))

    def record_gate(path=None, device='cuda'):
        gate_calls.append(str(device))
        value = json.loads(Path(path or reports / 'stage2a.json').read_text())
        assert value['status'] == 'PASS', 'Stage 2a did not pass'

    def tiny_stage1(*args, **kwargs):
        return dict(original_stage1(*args, **kwargs), pretrain_epochs=1, max_joint_steps=1)

    def tiny_paper(config):
        paper_config(config)
        config.update(pretrain_epochs=1, max_joint_steps=1)
        return config

    def tiny_mapper(*args):
        return dict(original_mapper(*args), epochs=1)

    stores = root / 'stores'
    store_log = []

    def fake_store_path(store, dest=None):
        return stores / store

    def fake_extract(store, device='cuda', limit=None, dest=None):
        store_log.append(('extract', store, device))
        (stores / store).mkdir(parents=True)
        extracted.add(store)
        return stores / store

    def fake_check(path):
        store_log.append(('check', Path(path).name))
        assert (Path(path) / 'ok').exists() or Path(path).name in extracted, 'fake store check failed'
        return {}

    class NotSmoke:
        """Diagnostics refuse weights whose sidecar says smoke; only the cpu smoke trains with --smoke."""

        def __getattr__(self, name):
            return getattr(json, name)

        @staticmethod
        def loads(text, *args, **kwargs):
            value = json.loads(text, *args, **kwargs)
            if isinstance(value, dict) and isinstance(value.get('config'), dict):
                value['config']['smoke'] = False
            return value

    extracted = set()
    original_stage1, original_mapper = pt._ORIGINAL_STAGE1_CONFIG, pt.mapper_config
    paper_config = paper.stage1_config

    def fake_s2a_check(root_, destination, *, device='cuda', causes=None, chunked=True, **kwargs):
        status = 'FAIL' if fake_s2a_check.fail else 'PASS'
        value = dict(schema=1, stage='2a', smoke=True, status=status, failed=['smoke'] if fake_s2a_check.fail else [],
                     missing=[], device=device)
        save_json(Path(destination), value)
        return value

    fake_s2a_check.fail = False
    s2a_calls = []

    def fake_dec_stage(name):
        return lambda *a, **k: s2a_calls.append(name)

    with ExitStack() as stack, threadpool_limits(limits=8):
        torch.set_num_threads(8)
        for module, name, value in [
                (ct2_data, 'split2', world.split), (ct2_data, 'default_images', world.default_images),
                (ct2_data, 'default_captions', world.default_captions),
                (ct2_data, 'instructed_images', world.instructed_images),
                (ct2_data, 'instructed_captions', world.instructed_captions),
                (pt, 'caption_features', world.caption_features), (pt, 'image_features', world.image_features),
                (pt, 'patch_tokens', lambda s, pids: world.patches(s['all_painting_ids'][pids])),
                (pt, '_ORIGINAL_STAGE1_CONFIG', tiny_stage1), (pt, 'mapper_config', tiny_mapper),
                (paper, 'stage1_config', tiny_paper),
                (pt, 'FROZEN_HELD_LIST', frozen_list), (ct2_verdict, 'FROZEN_HELD_LIST', frozen_list),
                (ct_held, 'held_names', lambda: list(world.held_names)),
                (ct_held, 'painting_labels', world.painting_labels),
                (ct_held, 'qwen_default_images', lambda names: world.qwen[world.rows(names)].copy()),
                (ct_held, 'clip_global_images', lambda names: world.clip[world.rows(names)].copy()),
                (ct_held, 'clip_patches', world.patches),
                (ct_diag, 'target_images', lambda aspect, names, sha, root=None: np.roll(
                    world.qwen[world.rows(names)], 1 + ('emotion', 'style', 'genre').index(aspect), axis=1)),
                (ct2_diag, 'FIT_ROWS', N_FIT),
                (ct2_stage2a, 'require_pass', record_gate),
                (ct2_stage2a, 'train', fake_dec_stage('train')), (ct2_stage2a, 'train_chunked', fake_dec_stage('chunked')),
                (ct2_stage2a, 'check', fake_s2a_check),
                (ct2_extract, 'store_path', fake_store_path), (ct2_extract, 'extract', fake_extract),
                (ct2_data, 'check_s2_store', fake_check),
                (ct2_verdict, 'paired_bootstrap', lambda *a, **k: sc.paired_bootstrap(
                    *a, **dict(k, n_resamples=8, workers=1))),
                (ct2_diag, 'run', lambda *a, **k: original_diag_run(*a, **dict(k, n_resamples=8)))]:
            if name == 'run' and module is ct2_diag:
                original_diag_run = ct2_diag.run
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.dict(ms_train.CONFIG, max_epochs=1))
        stack.enter_context(patch.dict(ct2_verdict.FIT_PAIRS, full=2000 if smoke else CAPTIONS * N_FIT))   # --smoke: first 2,000 fit captions
        stack.enter_context(patch.object(ct_diag, 'HERE', reports))
        if smoke:
            stack.enter_context(patch.object(ct2_diag, 'json', NotSmoke()))

        if 'budget' in steps:
            assert ct2_run.main(['budget']) == 0
            value = ct2_run.budget()
            assert value['measured'] is False and value['route'] in ('local', 'run chat: decide on node408')
            passed('budget')
        if 'extract' in steps:
            (stores / 'genre_word_img').mkdir(parents=True, exist_ok=True)
            (stores / 'genre_word_img/ok').write_text('')
            call('extract')
            assert [e for e in store_log if e[0] == 'extract'] == [
                ('extract', s, device) for s in ct2_extract.STORES if s != 'genre_word_img']
            assert ('check', 'genre_word_img') in store_log
            assert not any(e == ('extract', 'genre_word_img', device) for e in store_log)
            # An existing store that fails the check stops the step and is never touched.
            (stores / 'genre_word_img/ok').unlink()
            before_files = sorted(p.name for p in (stores / 'genre_word_img').iterdir())
            try:
                call('extract')
                raise SystemExit('extract accepted a failing store')
            except AssertionError:
                pass
            assert sorted(p.name for p in (stores / 'genre_word_img').iterdir()) == before_files
            passed('extract')
        if 'graphs' in steps:
            call('graphs')
            manifest = json.loads((full / 'd2/graphs/manifest.json').read_text())
            assert manifest['identity']['mode'] == 'full' and manifest['identity']['fit_size'] == N_FIT
            assert all((full / f'd2/graphs/{a}.npz').exists() for a in ('emotion', 'style', 'genre'))
            stamp = (full / 'd2/graphs/manifest.json').stat().st_mtime_ns
            call('graphs')   # an existing matching folder is skipped, not rebuilt
            assert (full / 'd2/graphs/manifest.json').stat().st_mtime_ns == stamp
            passed('graphs')
        if 'stage2a' in steps:
            fake_s2a_check.fail = True
            assert call('stage2a') == 1, 'a failing stage 2a must return a nonzero code'
            assert json.loads((reports / 'stage2a.json').read_text())['status'] == 'FAIL'
            fake_s2a_check.fail = False
            assert call('stage2a') == 0
            assert json.loads((reports / 'stage2a.json').read_text())['status'] == 'PASS'
            assert s2a_calls.count('train') == 2 and s2a_calls.count('chunked') == 2
            calls = len(s2a_calls)
            assert call('stage2a') == 0 and len(s2a_calls) == calls, 'a passing report must skip the step'
            passed('stage2a')
        if 'd2' in steps:
            call('d2')
            for seed in SEEDS:
                for aspect in ('emotion', 'style', 'genre'):
                    assert list((full / 'd2/train/D2').glob(f'{aspect}__*__s{seed}.npz')), (aspect, seed)
            assert (full / 'd2/weights').exists()
            passed('d2')
        for settings in ('core', 'paper'):
            if settings in steps:
                call(settings)
                for seed in SEEDS:
                    for version in pt.VERSIONS:
                        assert (full / f'percept/{settings}/stage1/{version}__s{seed}.npz').exists()
                    for cut in ct2_run.CUTS[settings]:
                        for arm in ct2_run.ARMS[settings]:
                            path = full / f'percept/{settings}/stage2/{cut}/{arm}__s{seed}.npz'
                            if not path.exists():
                                assert settings == 'paper' and cut == 'strict', path   # a single-topic cut
                assert not list((full / 'held').rglob('*.npz'))
                passed(settings)
        if 'rcol' in steps:
            call('rcol')
            rcol = json.loads((full / 'percept/rcol.json').read_text())
            assert {r['seed'] for r in rcol['runs']} == set(SEEDS)
            passed('rcol')
        if 'held' in steps:
            call('held')
            held = sorted(str(p.relative_to(full / 'held')) for p in (full / 'held').rglob('*.npz'))
            assert len([h for h in held if h.startswith('d2/')]) == 3 * len(SEEDS), held
            assert json.loads((full / 'held/held_paintings.json').read_text()) == json.loads(frozen_list.read_text())
            call('held')   # a rerun verifies and reuses
            passed('held')
        if 'verdict' in steps:
            call('verdict')
            verdict = json.loads((reports / 'verdict_stage2.json').read_text())
            assert verdict['stage'] == 2 and verdict['outcome_id'] in ('S2-GO', 'S2-MM', 'S2-IN', 'S2-NO')
            passed('verdict')
        if 'diagnostics' in steps:
            call('diagnostics')
            diag = json.loads((reports / 'diag_stage2.json').read_text())
            assert diag['diagnostic_only'] and diag['fit_rows'] == N_FIT and diag['records']
            passed('diagnostics')
        if 'gates' in steps:
            calls = len(gate_calls)
            # V4: with held output present every training step refuses before it trains.
            for step in ('d2', 'core', 'paper'):
                try:
                    call(step)
                    raise SystemExit(f'{step} trained after held output exists')
                except AssertionError as error:
                    assert 'V4' in str(error), error
            # V1: a failing stage 2a gate stops every later step before it reads anything.
            def refuse(path=None, device='cuda'):
                raise AssertionError('Stage 2a did not pass')
            with patch.object(ct2_stage2a, 'require_pass', refuse):
                for step in ct2_run.GATED:
                    try:
                        call(step)
                        raise SystemExit(f'{step} ran without the stage 2a pass')
                    except AssertionError as error:
                        assert 'Stage 2a' in str(error), error
            assert set(gate_calls[:calls]) <= {device}
            passed('gates')
        if set(steps) & set(ct2_run.GATED):
            assert set(gate_calls) == {device}, f'stage 2a gate devices: {set(gate_calls)}'
    after = production_state()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f'the smoke changed production paths: {changed[:5]}'
    report.update(status='PASS', seconds=time.perf_counter() - started, gate_calls=len(gate_calls))
    save_json(root / 'smoke.json', report)
    print(f'SMOKE ALL PASS seconds={report["seconds"]:.0f} gate_calls={len(gate_calls)}', flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scratch', type=Path)
    p.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    p.add_argument('--steps', default=','.join(STEPS), help='comma-separated subset of: ' + ', '.join(STEPS))
    args = p.parse_args(argv)
    run(args.scratch, args.device, args.steps.split(','))


if __name__ == '__main__':
    main()
