"""Train D2 ablations through the unchanged model-sheet cohort runner."""
import argparse
import copy
import hashlib
import json
import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from ab_paths import (ASPECT_BASE, MAIN_ROUND, MAIN_TEST, OUTPUTS, ROUND, SAMPLE,
                      check_imports, ms_train, pav_sample)
from ab_rows import ASPECTS, DEFAULT_SWITCHES, ROWS, resolve, validate

SMOKE_ROOT = Path(os.environ.get('AB_SMOKE_ROOT', '/tmp/claude-0'))
_ORIGINAL_HEAD_LOSS = ms_train.head_loss


def switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher=None,
                       extras=None, include_mask_penalty=True, *, switches=None):
    """Keep only the requested form-a terms, for both fit and dev calls."""
    switches = DEFAULT_SWITCHES if switches is None else switches
    if all(switches[k] == DEFAULT_SWITCHES[k] for k in ('buddy', 'own', 'endpoints')):
        return _ORIGINAL_HEAD_LOSS(models, img, cap, pairs, weights, row, aspect,
                                   teacher, extras, include_mask_penalty)
    assert row['form'] == 'a' and row['graph_terms'] and extras is None
    a, b = torch.as_tensor(pairs, device=img.device).unbind(1)
    ids = torch.cat((a, b))
    z = models['img'](img[ids])
    ia, ib = z.chunk(2)
    w = torch.as_tensor(weights, device=img.device)
    terms = []
    if switches['endpoints'] == 'image':
        if switches['buddy']:
            terms.append(ms_train.symmetric(ia, ib, w))
    else:
        ca, cb = models['cap'](cap[ids]).chunk(2)
        if switches['buddy']:
            terms.append((ms_train.symmetric(ia, ib, w) + ms_train.symmetric(ia, cb, w) +
                          ms_train.symmetric(ca, ib, w) + ms_train.symmetric(ca, cb, w)) / 4)
        if switches['own']:
            terms.append((ms_train.symmetric(ia, ca) + ms_train.symmetric(ib, cb)) / 2)
    if row['distill'] != 'none':
        terms.append(row['distill_weight'] * ms_train.distill_loss(z, teacher[ids], ids,
                                                                 row['distill']))
    assert terms, 'switches removed every loss term'
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def run_cohort(s, inputs, graphs, manifests, recipe, name, aspects, seed, device,
               outdir, smoke=False, max_epochs=200):
    """All-off rows enter run with no patch; switch patches unwind even on failure."""
    validate(recipe)
    switches = recipe['switches']
    with ExitStack() as stack:
        if switches != DEFAULT_SWITCHES:
            stack.enter_context(patch.object(ms_train, 'head_loss',
                                             partial(switched_head_loss, switches=switches)))
        if switches['teacher'] == 'default':
            def default_teacher(sample, key):
                assert sample is s, 'teacher sample differs from student sample'
                aspect = next(a for a in aspects if ms_train.head.ASPECTS[a] == key)
                return inputs[aspect]['img']
            stack.enter_context(patch.object(ms_train.ags_trainimg, 'aspect_images', default_teacher))
        return ms_train.run(s, inputs, graphs, manifests, recipe['base'], name, aspects,
                            seed, device, outdir, smoke, max_epochs)


def load_sample():
    """The worktree has code only; retain the frozen loader's positional checks."""
    with patch.object(pav_sample, 'SAMPLE', SAMPLE):
        return ms_train.head.load_frozen()


def require_graph_store(recipe):
    base = MAIN_ROUND if recipe['store'] == 'ab' else ASPECT_BASE
    if recipe['base']['graphs'] is not None:
        folder = base / 'graphs'
        required = ['manifest.json'] + [f'{a}.npz' for a in ASPECTS]
        missing = [str(folder / name) for name in required if not (folder / name).is_file()]
        assert not missing, f'missing graph store {folder}: {", ".join(missing)}'
    return base


def validate_outputs(outputs, smoke):
    outputs = Path(outputs).resolve()
    for test_root in (MAIN_TEST, ROUND.parent):
        for prefix in ('20261017_', '20261018_', '20261019_', '20261020_'):
            for earlier in test_root.glob(prefix + '*'):
                assert not outputs.is_relative_to(earlier.resolve()), 'outputs inside an earlier round'
    if smoke:
        assert outputs.is_relative_to(SMOKE_ROOT.resolve()), '--smoke outputs must be in the scratchpad'
    return outputs


def write_row_metadata(outputs, name, seeds):
    recipe = copy.deepcopy(ROWS[name])
    # The sheet's config cannot describe switches; bind them and this implementation here.
    code = hashlib.sha256(b''.join((ROUND / f).read_bytes()
                                  for f in ('ab_paths.py', 'ab_rows.py', 'ab_train.py'))).hexdigest()
    metadata = dict(recipe, name=name, code_sha256=code,
                    graph_store=str((MAIN_ROUND if recipe['store'] == 'ab' else ASPECT_BASE) / 'graphs'),
                    caption_side='not scored' if recipe['switches']['endpoints'] == 'image' else 'scored')
    path = Path(outputs) / 'train' / name / 'ab_row.json'
    graphs_by_seed = {}
    if path.exists():
        old = json.loads(path.read_text())
        graphs_by_seed = old.pop('graphs_by_seed')
        assert old == metadata, f'{path}: row metadata mismatch'
    else:
        assert not list(path.parent.glob('*.npz')), f'{path}: outputs lack row metadata'
    graphs_by_seed.update({str(seed): resolve(name, seed)['base']['graphs'] for seed in seeds})
    metadata['graphs_by_seed'] = graphs_by_seed
    ms_train.head.atomic_json(path, metadata)
    return metadata


@threadpool_limits.wrap(limits=8)
def train(rows, seeds=(0, 1, 2), device='cuda', outputs=OUTPUTS, *, smoke=False,
          max_epochs=None):
    rows, seeds = list(rows), list(seeds)
    assert rows and len(set(rows)) == len(rows) and all(r in ROWS for r in rows)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in (0, 1, 2) for s in seeds)
    max_epochs = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    assert not smoke or max_epochs == 2, '--smoke requires 2 epochs'
    assert 1 <= max_epochs <= ms_train.CONFIG['max_epochs']
    outputs = validate_outputs(outputs, smoke)
    check_imports()
    for name in rows:
        require_graph_store(resolve(name, seeds[0]))
    torch.set_num_threads(8)
    device = torch.device(device)
    assert device.type in ('cpu', 'cuda')
    s = load_sample()
    fit, dev = ms_train.disjoint_split(s)
    inputs = ms_train.head.load_inputs(s, ASPECTS[0], 'default')
    result = {}
    for name in rows:
        write_row_metadata(outputs, name, seeds)
        row = ROWS[name]['base']
        row_inputs = ({a: ms_train.head.load_inputs(s, a, 'prompted') for a in ASPECTS}
                      if row['inputs'] == 'instructed' else {a: inputs for a in ASPECTS})
        cohorts = [[a] for a in ASPECTS] if row['share'] == 'separate' else [list(ASPECTS)]
        result[name] = {}
        for seed in seeds:
            recipe = resolve(name, seed)
            base = require_graph_store(recipe)
            graphs, manifests = {}, {}
            for aspect in ASPECTS:
                if recipe['base']['graphs'] is None:
                    graphs[aspect] = dict(fit_node_ids=fit, dev_node_ids=dev)
                else:
                    graphs[aspect], manifests[aspect] = ms_train.head.load_graph(
                        s, base, aspect, [recipe['base']['graphs'][aspect]])
            result[name][seed] = {}
            for cohort in cohorts:
                trained = run_cohort(s, {a: row_inputs[a] for a in cohort}, graphs, manifests,
                                     recipe, name, cohort, seed, device, outputs / 'train',
                                     smoke, max_epochs)
                result[name][seed].update(trained)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', required=True, choices=list(ROWS))
    parser.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--outputs', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-epochs', type=int)
    args = parser.parse_args(argv)
    assert not args.smoke or args.outputs is not None, '--smoke requires explicit scratchpad --outputs'
    return train(args.rows, args.seeds, args.device,
                 args.outputs if args.outputs is not None else OUTPUTS,
                 smoke=args.smoke, max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()
