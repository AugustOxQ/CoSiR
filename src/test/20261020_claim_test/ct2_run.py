"""Stage 2 queue steps: one command per step, each calling the ct2 module CLIs unchanged.

Steps in queue order: extract, graphs, stage2a, d2, core, paper, rcol, held, then on CPU
verdict and diagnostics. Every step after stage2a first calls ct2_stage2a.require_pass (V1)
with the device given here. Seeds are fixed to 0 to 4. V4: once any stage 2 held array exists,
the training steps refuse to run. `--root` is the stage 2 output folder (s2a/ and full/ below
it); `--reports` holds stage2a.json, verdict_stage2.json and diag_stage2.json. `--smoke` is for
ct2_smoke.py only: it needs a root outside this round's outputs and passes --smoke to PercepT.
"""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(5))
STEPS = ('budget', 'extract', 'graphs', 'stage2a', 'd2', 'core', 'paper', 'rcol', 'held', 'verdict', 'diagnostics')
GATED = STEPS[STEPS.index('d2'):]
TRAINING = ('d2', 'core', 'paper')
CUTS = dict(core=('k40',), paper=('strict', 'k40'))
ARMS = dict(core=('P2', 'P3'), paper=('P1', 'P4'))   # full scale (stage 2a runs P1 to P4 at sample size)
# Unmeasured per-step estimates in minutes (low, high), from the ticket; k-means time is not included.
GPU_MINUTES = dict(stage2a=(25, 25), d2=(10, 10), core=(45, 45), paper_stage1=(85, 85), paper_stage2=(25, 25),
                   held=(5, 5))
CPU_MINUTES = dict(graphs=(3, 23), verdict=(5, 5), diagnostics=(35, 50))


def budget(now=None):
    """End estimate against the next 11:00 (Amsterdam); per-step numbers are estimates, not measurements."""
    now = now or datetime.now(ZoneInfo('Europe/Amsterdam'))
    low = sum(v[0] for v in (*GPU_MINUTES.values(), *CPU_MINUTES.values()))
    high = sum(v[1] for v in (*GPU_MINUTES.values(), *CPU_MINUTES.values()))
    deadline = now.replace(hour=11, minute=0, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    end_low, end_high = now + timedelta(minutes=low), now + timedelta(minutes=high)
    return dict(start=now.isoformat(), estimated_end_low=end_low.isoformat(), estimated_end_high=end_high.isoformat(),
                gpu_minutes=sum(v[1] for v in GPU_MINUTES.values()),
                cpu_minutes=dict(low=sum(v[0] for v in CPU_MINUTES.values()), high=sum(v[1] for v in CPU_MINUTES.values())),
                measured=False, excludes='k-means (one thread, n_init 10, 216,107 pairs) and the remaining extraction',
                deadline=deadline.isoformat(),
                route='run chat: decide on node408' if end_high > deadline else 'local')


def frozen_training(full):
    """V4: no training step once any stage 2 held array exists."""
    assert not any((Path(full) / 'held').rglob('*.npz')), 'V4: training refused, stage 2 held output exists'


def step_extract(device):
    """Extract only missing stores; an existing store must pass check_s2_store and is never touched."""
    import ct2_data
    import ct2_extract
    paths = {store: ct2_extract.store_path(store) for store in ct2_extract.STORES}
    for store, path in paths.items():
        if path.exists():
            ct2_data.check_s2_store(path)
            print(f'extract {store}: present and checked, skipped', flush=True)
    for store, path in paths.items():
        if not path.exists():
            print(f'extract {store}: missing, extracting', flush=True)
            ct2_extract.extract(store, device)
            ct2_data.check_s2_store(path)


def step_graphs(full):
    """Pre-build D2's graphs on CPU; an existing folder must match this mode's identity."""
    import ct2_d2
    import ct2_data
    import ct2_graphs
    out = Path(full) / 'd2/graphs'
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['schema'] == 1 and manifest['identity'] == ct2_d2._graph_identity(ct2_data.split2('full')), \
            f'{out}: existing graphs do not match the full split; not overwritten'
        print(f'graphs: {out} matches, skipped', flush=True)
        return
    with patch.object(sys, 'argv', ['ct2_graphs.py', '--mode', 'full', '--out', str(out)]):
        ct2_graphs.main()


def step_stage2a(root, report, device):
    import ct2_stage2a
    if Path(report).exists():
        try:
            ct2_stage2a.require_pass(report, device)
            print('stage2a: report already passes, skipped', flush=True)
            return 0
        except (AssertionError, OSError, ValueError, KeyError) as error:
            print(f'stage2a: existing report does not pass ({error}); running', flush=True)
    return ct2_stage2a.main(['--root', str(Path(root) / 's2a'), '--report', str(report), '--device', device])


def percept_args(mode_root, settings, stage, device, smoke, seeds):
    args = ['--mode', 'full', '--settings', settings, '--stage', stage, '--seeds', *map(str, seeds), '--root', str(mode_root)]
    if device is not None:
        args += ['--device', device]
    return args + (['--smoke'] if smoke else [])


def step_percept(full, settings, device, smoke, seeds):
    """Stage 1 for both versions, then every full-scale arm at the settings' cuts."""
    import ct2_percept
    for version in ct2_percept.VERSIONS:
        ct2_percept.main(percept_args(full, settings, '1', device, smoke, seeds) + ['--version', version])
    for cut in CUTS[settings]:
        for arm in ARMS[settings]:
            ct2_percept.main(percept_args(full, settings, '2', device, smoke, seeds) + ['--cut', cut, '--arm', arm])


def step_held(full, smoke, seeds):
    """D2 then PercepT held inference, each on the device its weights were trained on."""
    import ct_held
    import ct2_d2
    import ct2_percept
    held_list = Path(full) / 'held/held_paintings.json'
    if not held_list.exists():
        ct_held.write_held_list(held_list)
    ct_held.read_held_list(held_list)
    ct2_d2.main(['--mode', 'full', '--held', '--root', str(full), '--seeds', *map(str, seeds)])
    for settings in CUTS:
        for cut in CUTS[settings]:
            for arm in ARMS[settings]:
                ct2_percept.main(percept_args(full, settings, 'held', None, smoke, seeds) + ['--cut', cut, '--arm', arm])


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # determinism() needs it before CUDA work
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=STEPS)
    p.add_argument('--root', type=Path, default=HERE / 'outputs/stage2')
    p.add_argument('--reports', type=Path, default=HERE)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--replace', metavar='REASON', help='V4 V-check rerun reason for replacing a changed verdict')
    args = p.parse_args(argv)
    assert args.replace is None or args.step == 'verdict', '--replace applies to the verdict step only'
    if args.step == 'budget':
        print(json.dumps(budget(), sort_keys=True), flush=True)
        return 0
    root, reports, full = args.root.resolve(), args.reports.resolve(), args.root.resolve() / 'full'
    if args.smoke:
        assert not root.is_relative_to(HERE / 'outputs') and not reports.is_relative_to(HERE / 'outputs'), \
            '--smoke needs a root outside this round\'s outputs'
    else:
        assert args.device.startswith('cuda'), 'full runs bind the stage 2a pass to cuda'
    report, verdict, diag = reports / 'stage2a.json', reports / 'verdict_stage2.json', reports / 'diag_stage2.json'
    seeds = SEEDS
    if args.step == 'extract':
        step_extract(args.device)
    elif args.step == 'graphs':
        step_graphs(full)
    elif args.step == 'stage2a':
        code = step_stage2a(root, report, args.device)
        if code:
            return code
    else:
        import ct2_stage2a
        ct2_stage2a.require_pass(report, args.device)
        if args.step in TRAINING:
            frozen_training(full)
        if args.step == 'd2':
            import ct2_d2
            ct2_d2.main(['--mode', 'full', '--train', '--device', args.device, '--root', str(full), '--seeds', *map(str, seeds)])
        elif args.step in ('core', 'paper'):
            step_percept(full, args.step, args.device, args.smoke, seeds)
        elif args.step == 'rcol':
            import ct2_verdict
            ct2_verdict.main(['--rcol-only', '--mode', 'full', '--root', str(full)])
        elif args.step == 'held':
            step_held(full, args.smoke, seeds)
        elif args.step == 'verdict':
            import ct2_verdict
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=8):
                ct2_verdict.main(['--root', str(full), '--stage2a', str(report), '--device', args.device,
                                  '--verdict', str(verdict), '--workers', '8'] +
                                 (['--replace', args.replace] if args.replace else []))
        elif args.step == 'diagnostics':
            import ct2_diag
            ct2_diag.main(['--verdict', str(verdict), '--root', str(full), '--out', str(diag), '--workers', '8',
                          '--stage2a', str(report), '--device', args.device])
    print(f'CT2_RUN {args.step} OK', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
