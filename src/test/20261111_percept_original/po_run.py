"""r26 PercepT on CLIP L/14 + ModernBERT-GoEmotions: the four steps, each over all seeds and both cuts, idempotent.

    python po_run.py stage1 [--seeds 0 1 2 3 4]      Stage 1 (both cuts in one artefact per seed); GPU
    python po_run.py stage2 [--cuts k40 strict]      Stage 2 attention pooling on L/14 patch tokens; GPU (17 GB)
    python po_run.py held                            image-only inference on the 12,281 held paintings; GPU
    python po_run.py score [--record]                collapse check and AMI per cut/reading/aspect; CPU

A finished, verified output is skipped. `--smoke` needs PO_OUTPUTS=<scratch dir> (never the default outputs): one seed
(0), 2,000 fit pairs and 2 + 5 epochs in Stage 1, 200 fit paintings and 3 epochs in Stage 2; the other steps run as usual
on those smoke outputs. `score --smoke` never writes records/. The controller launches each GPU step through the queue.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import po_paths as P


def line(step, what, seconds):
    print(f'TIMING {step} {what}: {seconds:.1f} s', flush=True)


def step_stage1(po, s, seeds, device, smoke):
    for seed in seeds:
        t = time.perf_counter()
        if po.complete_stage1(s, P.OUT, seed, smoke):
            print(f'Verified complete: {po.stage1_path(P.OUT, seed)}', flush=True)
        else:
            po.run_stage1(s, P.OUT, seed, device=device, smoke=smoke)
        line('stage1', f'seed={seed}', time.perf_counter() - t)


def step_stage2(po, s, seeds, cuts, device, smoke):
    import torch
    sample_sha = po.pc_data.fingerprint(s)
    store, fit_tokens = po.PatchStore(), None
    for seed in seeds:
        for cut in cuts:
            t = time.perf_counter()
            if po.verify_stage1(P.OUT, seed)['K_by_cut'][cut] < 2:
                print(f'Skipping {cut} seed={seed}: fewer than 2 topics', flush=True)
            elif po.complete_stage2(P.OUT, cut, seed, smoke, sample_sha):
                print(f'Verified complete: {po.stage2_path(P.OUT, cut, seed)}', flush=True)
            else:
                if fit_tokens is None:
                    fit_tokens = po.load_fit_tokens(s, store, device or 'cuda', smoke)
                po.run_stage2(s, P.OUT, cut, seed, device=device, smoke=smoke, fit_tokens=fit_tokens, store=store)
            line('stage2', f'{cut} seed={seed}', time.perf_counter() - t)
    del fit_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def step_held(po, seeds, cuts, device, smoke):
    names, store = po.held_names(), po.PatchStore()
    for seed in seeds:
        for cut in cuts:
            t = time.perf_counter()
            po.run_held(P.OUT, cut, seed, names, device=device, smoke=smoke, store=store)
            line('held', f'{cut} seed={seed}', time.perf_counter() - t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('stage1', 'stage2', 'held', 'score'))
    ap.add_argument('--seeds', nargs='+', type=int, choices=P.SEEDS)
    ap.add_argument('--cuts', nargs='+', choices=P.CUTS, default=list(P.CUTS))
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--record', action='store_true', help='score only: also write the committed copies under records/')
    a = ap.parse_args(argv)
    if a.smoke and P.OUT.resolve() == P.DEFAULT_OUT.resolve():
        ap.error('--smoke requires PO_OUTPUTS=<scratch dir>')
    if a.smoke and a.record:
        ap.error('--smoke never writes records/')
    seeds = a.seeds if a.seeds is not None else ([0] if a.smoke else list(P.SEEDS))
    if len(set(seeds)) != len(seeds):
        ap.error('duplicate seeds')
    start = time.perf_counter()
    if a.step == 'score':
        import po_score
        args = ['--seeds', *map(str, seeds), '--cuts', *a.cuts] + (['--record'] if a.record else []) + \
               (['--smoke'] if a.smoke else [])
        po_score.main(args)
    else:
        import po_percept as po
        s = po.split()
        if a.step == 'stage1':
            step_stage1(po, s, seeds, a.device, a.smoke)
        elif a.step == 'stage2':
            step_stage2(po, s, seeds, a.cuts, a.device, a.smoke)
        else:
            step_held(po, seeds, a.cuts, a.device, a.smoke)
    line(a.step, 'total', time.perf_counter() - start)


if __name__ == '__main__':
    main()
