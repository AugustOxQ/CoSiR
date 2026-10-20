"""Queue adapters: explicit local write paths around the frozen ticket modules."""
import argparse
import json
import os
from pathlib import Path

import ct_stage0 as gate

HERE = Path(__file__).resolve().parent


def configure(outputs):
    """Redirect artifact writers; feature and frozen-sample readers stay unchanged."""
    import ct_d2
    import ct_percept
    import ct_rcol
    import ct_score
    outputs = Path(outputs).resolve()
    assert outputs.is_relative_to(HERE), 'full-run outputs must stay in this round worktree'
    ct_d2.OUTPUTS = ct_percept.OUTPUTS = ct_score.OUTPUTS = outputs
    ct_rcol.STAGE1 = outputs / 'percept/paper/stage1'
    ct_rcol.OUTPUT = outputs / 'percept/rcol_stage1.json'
    return outputs


STAGE0_ARTIFACTS = dict(d2=('d2/train/D2', 'd2/weights'),
                        core=('percept/core/stage1', 'percept/core/stage2/k40'))


def prepare_stage0_training(outputs, step):
    """Seed 0 artifacts must come from the code that is running now (V1).

    A step records the code identity it trains under in outputs/stage0_training_code.json.
    Seed 0 artifacts left by a different (or unrecorded) identity are renamed into
    outputs/stale/<Amsterdam timestamp>/, never deleted, so the step retrains them.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    outputs = Path(outputs)
    current = gate.code_identity()
    if gate.read_training_code(outputs).get(step) == current:
        return []
    old = [path for folder in STAGE0_ARTIFACTS[step] for path in sorted((outputs / folder).glob('*__s0.*'))]
    if old:
        stamp = datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y%m%d_%H%M%S')
        for path in old:
            target = outputs / 'stale' / stamp / path.relative_to(outputs)
            assert not target.exists(), f'stale target exists: {target}'
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
    gate.write_training_code(outputs, step, current)
    return old


def frozen_training(outputs, device='cuda'):
    """V4: once held outputs exist, every training artifact must remain complete."""
    import ct_d2
    import ct_percept as pt
    outputs = Path(outputs)
    if not any((outputs / 'held').rglob('*.npz')):
        return
    for seed in range(5):
        for aspect in gate.ASPECTS:
            path = outputs / 'd2/train/D2' / f'{aspect}__{gate.GRAPHS[aspect]}__s{seed}.npz'
            meta = ct_d2._artifact_info(path, 'output')
            weights = ct_d2._weights_info(outputs, aspect, seed)
            assert meta['config'] == weights['config'] and meta['config']['device'] == device
            assert meta['output_sha256'] == weights['output_sha256']
        for settings in ('core', 'paper'):
            for version in pt.VERSIONS:
                source = pt.verify_output(pt.stage1_path(settings, version, seed))
                assert not source['config']['smoke'] and source['config']['device'] == device
            cuts = ('k40',) if settings == 'core' else ('strict', 'k40')
            arms = pt.ARMS if settings == 'core' else ('P1', 'P4')
            for cut in cuts:
                for arm in arms:
                    source = pt.verify_output(pt.stage1_path(settings, pt.ARMS[arm][0], seed))
                    if settings == 'paper' and source['K_by_cut'][cut] < 2:
                        continue
                    meta = pt.verify_output(pt.stage2_path(settings, cut, arm, seed))
                    assert not meta['config']['smoke'] and meta['config']['device'] == device
                    gate.verify_dependency(meta, source, gate.file_sha256(
                        pt.stage1_path(settings, pt.ARMS[arm][0], seed)))


def frozen_held(outputs, device='cuda'):
    """V4: validate existing held arrays before a writer could replace one."""
    import numpy as np
    import ct_held
    import ct_d2
    import ct_percept as pt
    outputs = Path(outputs)
    names, sha = ct_held.read_held_list(outputs / 'held/held_paintings.json')
    for path in (outputs / 'held/d2').glob('*.npz'):
        aspect, seed = path.stem.split('__s')
        source = ct_d2._weights_info(outputs, aspect, int(seed))
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['held_sha256'] == sha and meta['rows'] == len(names)
        assert meta['weights_sha256'] == source['weights_sha256']
        assert gate.file_sha256(path) == meta['output_sha256'], 'V4: existing D2 held output changed'
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['img'] and z['img'].shape == (len(names), 2048)
            assert z['img'].dtype == np.float16 and np.isfinite(z['img']).all()
    nonfinite = set()
    for path in (outputs / 'held/percept').glob('*/*.npz'):
        settings, cut = path.parent.name.split('__')
        arm, seed = path.stem.split('__s')
        source = pt.verify_output(pt.stage2_path(settings, cut, arm, int(seed)))
        if settings == 'core':
            pt.verify_held_output(path, source, names, sha, device)
            continue
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['config'] == source['config'] and meta['inference_device'] == device
        assert meta['weights_sha256'] == source['weights_sha256']
        assert meta['held_sha256'] == sha and meta['rows'] == len(names)
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['prob']
            prob = z['prob']
            assert prob.shape == (len(names), source['K']) and prob.dtype == np.float32
            assert meta['array_sha256s'] == dict(prob=pt.pc_stage1.array_sha256(prob)), \
                'V4: existing paper held output changed'
            if not np.isfinite(prob).all():
                nonfinite.add((settings, cut, arm, int(seed)))
            else:
                assert ((prob >= 0) & (prob <= 1)).all()
    return nonfinite


def train_percept(settings, seeds, device):
    import ct_percept as pt
    for version in pt.VERSIONS:
        pt.main(['--settings', settings, '--stage', '1', '--version', version,
                 '--seeds', *map(str, seeds), '--device', device])
    for cut in (('k40',) if settings == 'core' else ('strict', 'k40')):
        for arm in (pt.ARMS if settings == 'core' else ('P1', 'P4')):
            pt.main(['--settings', settings, '--stage', '2', '--cut', cut, '--arm', arm,
                     '--seeds', *map(str, seeds), '--device', device])


def budget(gpu_hours=3., cpu_hours=1.5):
    """Print the local end estimate at submission; the run chat handles node408."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Europe/Amsterdam'))
    end = now + timedelta(hours=gpu_hours + cpu_hours)
    deadline = now.replace(hour=11, minute=0, second=0, microsecond=0)
    return dict(start=now.isoformat(), estimated_end=end.isoformat(),
                gpu_hours=gpu_hours, cpu_hours=cpu_hours,
                route='run chat: move PercepT to node408' if end > deadline else 'local',
                deadline=deadline.isoformat())


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # determinism() needs it before CUDA work
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=('budget', 'frozen', 'held-list', 'd2', 'core', 'paper',
                                   'rcol', 'held', 'images', 'verdict', 'diagnostics'))
    p.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    p.add_argument('--report', type=Path, default=HERE / 'stage0.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--stage0-training', action='store_true')
    args = p.parse_args(argv)
    if args.step == 'budget':
        print(json.dumps(budget(), sort_keys=True), flush=True)
        return
    assert args.device.startswith('cuda'), 'full runs require GPU training'
    outputs = configure(args.outputs)
    if args.stage0_training:
        assert args.step in ('d2', 'core')
        assert not any((outputs / 'held').rglob('*.npz')), 'V4: Stage 0 retraining after held output exists'
        prepare_stage0_training(outputs, args.step)
    else:
        gate.require_pass(args.report, args.device)
    if args.step in ('frozen', 'd2', 'core', 'paper'):
        frozen_training(outputs, args.device)
    seeds = [0] if args.stage0_training else list(range(5))
    if args.step == 'frozen':
        return
    if args.step == 'held-list':
        import ct_held
        ct_held.write_held_list(outputs / 'held/held_paintings.json')
    elif args.step == 'd2':
        import ct_d2
        ct_d2.train(seeds, args.device, outputs)
    elif args.step in ('core', 'paper'):
        train_percept(args.step, seeds, args.device)
    elif args.step == 'rcol':
        import ct_rcol
        ct_rcol.write_stage1(ct_rcol.STAGE1, ct_rcol.OUTPUT)
    elif args.step == 'held':
        nonfinite = frozen_held(outputs, args.device)
        import ct_d2
        import ct_percept as pt
        ct_d2.write_held(seeds, outdir=outputs)
        for settings in ('core', 'paper'):
            for cut in (('k40',) if settings == 'core' else ('strict', 'k40')):
                for arm in (pt.ARMS if settings == 'core' else ('P1', 'P4')):
                    for seed in seeds:
                        if (settings, cut, arm, seed) in nonfinite:
                            print(f'V4: preserving nonfinite {settings}/{cut}/{arm}/s{seed} for collapse', flush=True)
                            continue
                        pt.run_held(settings, cut, arm, seed)
    elif args.step == 'images':
        import ct_extract
        ct_extract.HELD_LIST = outputs / 'held/held_paintings.json'
        ct_extract.main(['--device', args.device])
    elif args.step == 'verdict':
        import subprocess
        import ct_held
        import ct_rcol
        import ct_score as sc
        rcol, eligible, widths = sc.read_rcol(ct_rcol.OUTPUT)
        names, sha = ct_held.read_held_list(outputs / 'held/held_paintings.json')
        assert names == ct_held.held_names()
        labels = ct_held.painting_labels(names)
        fixed, eligibility, sources = sc.prepare_held(names, sha, eligible, widths, output_root=outputs)
        result = sc.paired_bootstrap(fixed, labels['emotion'], eligibility['eligible_seeds'], workers=8)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
        sc.write_verdict(HERE / 'verdict_stage1.json', result, eligibility, sha, labels, commit,
                         dict(held_outputs=sources, rcol=dict(path=str(ct_rcol.OUTPUT),
                              sha256=gate.file_sha256(ct_rcol.OUTPUT), record=rcol)))
    elif args.step == 'diagnostics':
        import ct_diag
        ct_diag.run(HERE / 'verdict_stage1.json', outputs, HERE / 'diag_stage1.json', workers=8)


if __name__ == '__main__':
    main()
