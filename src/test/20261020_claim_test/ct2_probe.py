"""JSON-line GPU memory probe for full-scale PercepT, using synthetic inputs.

Includes one core joint DEC epoch (initial/post-update assignments included),
original and checkpointed, and one epoch plus dev/eval inference for each mapper.
Paper strict uses 100 topics: its maximum possible width before the cut is known.
Peaks include resident inputs, model, gradients and Adam states. No data is read
and no artifacts are written. CPU --smoke exercises the workloads without CUDA.
"""
import argparse
from contextlib import redirect_stdout
import gc
import json
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

import ct2_percept as ct
import ct_paper_sept as paper

FIT_PAINTINGS = 42969
FIT_PAIRS = 216107
DEV_PAINTINGS = 6152


def probe_device(device):
    device = torch.device(device)
    return torch.device('cuda:0') if device.type == 'cuda' and device.index is None else device


def probe_items():
    items = [dict(stage='joint', settings='core', version=version, chunked=chunked, topics=100)
             for version in ct.VERSIONS for chunked in (True, False)]
    items += [dict(stage='mapper', settings='core', arm=arm, cut='k40', topics=40)
              for arm in ('P2', 'P3')]
    items += [dict(stage='mapper', settings='paper', arm=arm, cut=cut, topics=k)
              for cut, k in (('strict', 100), ('k40', 40)) for arm in ('P1', 'P4')]
    return items


def measure(item, device, work):
    """Measure a workload; only CUDA OOM becomes a reported result."""
    device = probe_device(device)
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    result = dict(item, device=str(device), status='ok')
    try:
        # Training logs must never contaminate the JSON-line stdout stream.
        with redirect_stdout(sys.stderr):
            work()
        if cuda:
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='out_of_memory', error=str(error))
    result.update(seconds=time.perf_counter() - tick,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None)
    if cuda:
        result['gpu_name'] = torch.cuda.get_device_name(device)
    return result


def joint_epoch(item, device, fit_pairs):
    """Call the actual full-batch core loop, changing only its epoch ceiling."""
    width = ct.pc_stage1.VERSIONS[item['version']][3]
    inputs = torch.randn(fit_pairs, width)
    encoder, decoder = ct.pc_sept.build_autoencoder(width)
    encoder.to(device)
    decoder.to(device)
    centers = torch.nn.Parameter(torch.randn(100, 128, device=device))
    assignments = ct.chunked_soft_assignments if item['chunked'] else ct.ORIGINAL_SOFT_ASSIGNMENTS
    with patch.object(ct.pc_sept, 'MAX_DEC_EPOCHS', 1), \
            patch.object(ct.pc_sept.base, 'soft_assignments', assignments):
        ct.pc_sept.train_dec_until_stable(encoder, decoder, centers, inputs, device, print, 100)


def mapper_epoch(item, device, fit_paintings, dev_paintings):
    """Use the real mapper recipes and their resident fit/dev/eval tensors."""
    arm, settings, k = item['arm'], item['settings'], item['topics']
    s = dict(mode='full', first=np.arange(fit_paintings + dev_paintings),
             fit=np.arange(fit_paintings, dtype=np.int64),
             dev=np.arange(fit_paintings, fit_paintings + dev_paintings, dtype=np.int64))
    config = ct.mapper_config(s, arm, 0, False, device)
    config.update(settings=settings, cut=item['cut'], epochs=1)
    if settings == 'paper':
        paper.stage2_config(config)
    width = config['input_dim']
    shape = (50, width) if config['mapper'] == 'attention' else (width,)
    inputs = {}
    for part, rows in (('fit', fit_paintings), ('dev', dev_paintings), ('eval', dev_paintings)):
        # As image_inputs, allocate dev and eval independently even though full
        # mode evaluates validation twice. Patches retain stored normalization.
        x = torch.randn(rows, *shape, device=device)
        if len(shape) == 1:
            x.div_(torch.linalg.vector_norm(x, dim=1, keepdim=True))
        inputs[part] = x
    targets = dict(fit=torch.randint(0, 2, (fit_paintings, k)).numpy().astype(np.float32))
    trainer = paper.train_mapper if settings == 'paper' else ct.base.train_mapper_with_state
    trainer(inputs, targets, config, k)


def run_item(item, device='cuda', *, fit_pairs=FIT_PAIRS,
             fit_paintings=FIT_PAINTINGS, dev_paintings=DEV_PAINTINGS):
    """Run one independently cleaned probe item; counts can shrink for CPU tests."""
    device = probe_device(device)
    ct.base.determinism(0)
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    def work():
        if item['stage'] == 'joint':
            joint_epoch(item, device, fit_pairs)
        else:
            assert item['stage'] == 'mapper'
            mapper_epoch(item, device, fit_paintings, dev_paintings)
    try:
        result = measure(item, device, work)
        result.update(epochs=1, fit_pairs=fit_pairs, fit_paintings=fit_paintings,
                      dev_paintings=dev_paintings, eval_paintings=dev_paintings,
                      dec_chunk_rows=ct.DEC_CHUNK_ROWS if item.get('chunked') else None)
        return result
    finally:
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true', help='small inputs; CPU permitted')
    args = parser.parse_args(argv)
    assert args.smoke or torch.device(args.device).type == 'cuda', 'full probe requires CUDA'
    if torch.device(args.device).type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable'
    counts = dict(fit_pairs=37, fit_paintings=7, dev_paintings=3) if args.smoke else {}
    results = []
    for item in probe_items():
        result = run_item(item, args.device, **counts)
        result['smoke'] = args.smoke
        print(json.dumps(result, allow_nan=False), flush=True)
        results.append(result)
    return results


if __name__ == '__main__':
    main()
