"""Two-seed synthetic contract smoke, isolated from all claim artifacts and caches.

Real D2/core/paper optimizers, saved weights, image replay, cuts, scorer and
Leiden run with one epoch. Synthetic feature readers and a deterministic image
encoder replace external stores/model downloads. Fixed 5,000-row serializers
remain intact; only the smoke held list is tiny. This is not a Stage 0 pass.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import ct_stage0 as gate

HERE = Path(__file__).resolve().parent


def scratch_root(path):
    """Never let a smoke write to production artifacts or a shared feature store."""
    path = Path(path).resolve()
    forbidden = [HERE / 'outputs', Path('/project/CoSiR/src/test/20261020_claim_test'),
                 Path('/data')]
    assert not any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in forbidden), \
        'smoke needs a separate scratch output root'
    assert not path.exists(), 'smoke output root already exists; use a new scratch directory'
    return path


def sample(n, eval_start):
    rows = np.arange(n, dtype=np.int64)
    names = np.asarray([f'synthetic-{n}-{i:05d}' for i in rows])
    evaluation = np.arange(eval_start, eval_start + 5000, dtype=np.int64)
    return dict(pid=rows, first=rows, sample_ids=rows, painting_ids=names,
                all_painting_ids=names, subset_pid=evaluation,
                subset_image_indices=evaluation, subset_painting_ids=names[evaluation])


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(root, device='cuda'):
    """Run the contract chain on seeds 0 and 1; return its scratch smoke manifest."""
    import time
    started = time.perf_counter()
    import ct_d2 as d2
    import ct_percept as pt
    import ct_paper_sept as paper
    import ct_rcol
    import ct_held
    import ct_score as sc
    import ct_diag
    import ct_extract

    root = scratch_root(root)
    root.mkdir(parents=True)
    outputs = root / 'outputs'
    rng = np.random.default_rng(91)
    held_names = [f'synthetic-held-{i:03d}' for i in range(64)]
    sha = ct_held.names_sha256(held_names)
    list_path = outputs / 'held/held_paintings.json'
    save_json(list_path, dict(names=held_names, sha256=sha))
    report = dict(smoke=True, device=device, seeds=[0, 1], epochs=1, steps=[])

    def step(name):
        report['steps'].append(name)
        save_json(root / 'smoke.json', report)
        print(f'SMOKE {name} PASS', flush=True)

    step('held_list')
    ds, ps = sample(19889, 14889), sample(7004, 2004)
    all_names = ds['all_painting_ids'].tolist() + ps['all_painting_ids'].tolist() + held_names
    lookup = {name: i for i, name in enumerate(all_names)}
    # Structured variation avoids degenerate k-means while keeping store fixtures cheap.
    qwen = rng.normal(size=(len(all_names), 2048)).astype(np.float32)
    qwen /= np.linalg.norm(qwen, axis=1, keepdims=True)
    clip = rng.normal(size=(len(all_names), 512)).astype(np.float32)
    emotion = rng.normal(size=(7004, 768)).astype(np.float32)

    def rows(names):
        return np.asarray([lookup[name] for name in names], dtype=np.int64)

    def qimages(names):
        return qwen[rows(names)].copy()

    def cimages(names):
        x = clip[rows(names)].copy()
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def patches(names):
        return np.broadcast_to(cimages(names)[:, None, :], (len(names), 50, 512)).copy()

    fit, dev = d2.ms_train.disjoint_split(ds)
    # All aspects share the same input store, as in the production D2 path.
    shared = dict(img=qimages(ds['all_painting_ids']), cap=qimages(ds['all_painting_ids']))
    inputs = dict.fromkeys(gate.ASPECTS, shared)
    graphs, manifests = {}, {}
    for aspect in gate.ASPECTS:
        row = gate.GRAPHS[aspect]
        edges = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
        graph_rows = {part: (edges.copy(), np.ones(4, dtype=np.float32)) for part in ('fit', 'dev')}
        graphs[aspect] = dict(fit_node_ids=fit, dev_node_ids=dev, rows={row: graph_rows})
        manifests[aspect] = dict(identity={'method': 'synthetic smoke'}, archives={aspect: 'synthetic'},
                                 counts={aspect: {part: {row: 4} for part in ('fit', 'dev')}})
    partition = np.asarray(['fit'] * 2000 + ['dev'] * 4 + ['eval'] * 5000)
    original_run = d2.ms_train.run
    stage1_config, stage2_config = pt.pc_stage1.make_config, pt.pc_stage2.make_config
    paper_config = paper.stage1_config

    def tiny_d2(*args, **kwargs):
        return original_run(*args, **dict(kwargs, max_epochs=1))

    def tiny_stage1(*args, **kwargs):
        return dict(stage1_config(*args, **kwargs), pretrain_epochs=1, max_joint_steps=1)

    def tiny_stage2(*args, **kwargs):
        return dict(stage2_config(*args, **kwargs), epochs=1)

    def tiny_paper(config):
        paper_config(config)
        config.update(pretrain_epochs=1, max_joint_steps=1)
        return config

    def read_list(path):
        value = json.loads(Path(path).read_text())
        assert value['names'] == held_names and value['sha256'] == sha
        return value['names'], value['sha256']

    with ExitStack() as stack, threadpool_limits(limits=8):
        torch.set_num_threads(8)
        for module, name, value in [
                (d2, 'load_training_data', lambda: (ds, inputs, graphs, manifests)),
                (d2.ms_train, 'run', tiny_d2),
                (d2.ms_train.ags_trainimg, 'aspect_images', lambda s, key: inputs['emotion']['img']),
                (ct_held, 'qwen_default_images', qimages), (ct_held, 'clip_global_images', cimages),
                (ct_held, 'clip_patches', patches), (ct_held, 'read_held_list', read_list),
                (pt, 'OUTPUTS', outputs), (sc, 'HELD_ROWS', 64), (sc, 'SEEDS', [0, 1]),
                (pt.pc_data, 'partitions', lambda s: partition),
                (pt.pc_data, 'image_features', lambda s, kind:
                 qimages(s['all_painting_ids']) if kind == 'qwen_default' else clip[rows(s['all_painting_ids'])].copy()),
                (pt.pc_data, 'caption_features', lambda s, kind:
                 emotion.copy() if kind == 'roberta' else
                 qimages(s['painting_ids']) if kind.startswith('qwen_') else cimages(s['painting_ids'])),
                (pt.pc_data, 'patch_tokens', lambda s, ids: patches(s['all_painting_ids'][ids])),
                (pt.pc_stage1, 'make_config', tiny_stage1), (pt.pc_stage2, 'make_config', tiny_stage2),
                (paper, 'stage1_config', tiny_paper)]:
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.dict(d2.ms_train.CONFIG, max_epochs=1))
        d2.train([0, 1], device, outputs)
        for seed in (0, 1):
            for aspect in gate.ASPECTS:
                path = outputs / 'd2/train/D2' / f'{aspect}__{gate.GRAPHS[aspect]}__s{seed}.npz'
                with np.load(path) as z:
                    assert set(z.files) == {'img', 'cap'}
                    expected = z['img']
                assert gate.compare_replay(d2.infer(aspect, seed, ds['subset_painting_ids'].tolist(),
                                                   outdir=outputs), expected)['status'] == 'PASS'
        step('d2')
        for settings in ('core', 'paper'):
            for seed in (0, 1):
                for version in pt.VERSIONS:
                    path = pt.run_stage1(ps, settings, version, seed, smoke=True, device=device)
                    pt.verify_output(path)
                    with np.load(path) as z:
                        assert set(z.files) == pt.STAGE1_KEYS[settings]
                for cut in (('k40',) if settings == 'core' else ('strict', 'k40')):
                    for arm in (pt.ARMS if settings == 'core' else ('P1', 'P4')):
                        path = pt.run_stage2(ps, settings, cut, arm, seed, smoke=True, device=device)
                        if path is None:
                            source = pt.verify_output(pt.stage1_path(settings, pt.ARMS[arm][0], seed))
                            assert settings == 'paper' and source['K_by_cut'][cut] == 1
                            continue
                        pt.verify_output(path)
                        with np.load(path) as z:
                            assert z.files == ['eval_prob']
                            expected = z['eval_prob']
                        replay = pt.infer_stage2(settings, cut, arm, seed, ps['subset_painting_ids'].tolist())
                        assert gate.compare_replay(replay, expected)['status'] == 'PASS'
            step(settings)
        rcol_runs = []
        for seed in (0, 1):
            for version in pt.VERSIONS:
                with np.load(pt.stage1_path('paper', version, seed)) as z:
                    assert np.array_equal(z['kept_strict'], ct_rcol.strict_mask(z['center_norms']))
                    assert np.array_equal(z['kept_k40'], ct_rcol.k40_mask(z['fit_counts']))
                    for cut in ('strict', 'k40'):
                        judgment = ct_rcol.judge(cut, z[f'kept_{cut}'], z[f'fit_topics_{cut}'])
                        rcol_runs.append(dict(seed=seed, version=version, cut=cut, **judgment))
        entries, widths = {}, {}
        for entry, version, cut in [('B2', 'clip_roberta', 'strict'), ('B3', 'qwen_qwen', 'strict'),
                                    ('B4', 'clip_roberta', 'k40'), ('B5', 'qwen_qwen', 'k40')]:
            runs = [r for r in rcol_runs if r['version'] == version and r['cut'] == cut]
            seeds = [r['seed'] for r in runs if not r['collapsed']]
            entries[entry] = dict(eligible=bool(seeds), seeds=seeds)
            if cut == 'strict':
                widths[entry] = {r['seed']: r['kept_count'] for r in runs}
        rcol = dict(smoke=True, written='synthetic smoke', entries=entries, runs=rcol_runs)
        rcol_path = outputs / 'percept/rcol_stage1.json'
        save_json(rcol_path, rcol)
        _, eligible, strict_widths = sc.read_rcol(rcol_path)
        assert strict_widths == widths
        step('rcol')
        d2.write_held([0, 1], outdir=outputs)
        for settings in ('core', 'paper'):
            for cut in (('k40',) if settings == 'core' else ('strict', 'k40')):
                for arm in (pt.ARMS if settings == 'core' else ('P1', 'P4')):
                    for seed in (0, 1):
                        source_path = pt.stage2_path(settings, cut, arm, seed)
                        if not source_path.exists():
                            continue
                        source = pt.verify_output(source_path)
                        prob = pt.infer_stage2(settings, cut, arm, seed, held_names)
                        path = outputs / 'held/percept' / f'{settings}__{cut}' / f'{arm}__s{seed}.npz'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, prob=prob)
                        save_json(path.with_suffix('.json'), dict(held_sha256=sha, rows=64,
                                  weights_sha256=source['weights_sha256'], config=source['config']))
        step('held_inference')
        class Image:
            def __init__(self, row):
                self.row = row

            def close(self):
                pass

        class Encoder:
            m = SimpleNamespace(config=SimpleNamespace(_commit_hash='f' * 40))

            def encode_images(self, images, batch_size):
                return np.eye(2048, dtype=np.float32)[[im.row for im in images]]

        spec = SimpleNamespace(images=[dict(painting=n) for n in held_names], load_image=Image)
        enc = Encoder()
        for key in ct_extract.KEYS:
            enc.instruction = ct_extract.PROMPTS[key]
            path = ct_extract.write_store(enc, key, held_names, sha, spec, np.arange(64),
                                          root=outputs / 'targets')
            ct_extract.verify_store(path, key, held_names, sha)
        step('instructed_images')
        labels = {a: np.arange(64, dtype=np.int64) % n for a, n in
                  [('emotion', 9), ('style', 8), ('genre', 4)]}
        fixed, eligibility, sources = sc.prepare_held(held_names, sha, eligible, widths, output_root=outputs)
        result = sc.paired_bootstrap(fixed, labels['emotion'], eligibility['eligible_seeds'],
                                     n_resamples=8, workers=1)
        # Never use the production verdict filename, even in the scratch directory.
        verdict = sc.write_verdict(root / 'smoke_verdict.json', result, eligibility, sha, labels,
                                    'synthetic-smoke', dict(held_outputs=sources, rcol=rcol))
        assert {'d', 'interval', 'A_e', 'outcome_id', 'held_sha256'} <= set(verdict)
        ct_diag.read_verdict(root / 'smoke_verdict.json')
        step('verdict')
        diagnostics = []
        for arm, runs in fixed.items():
            for seed, partitions in runs.items():
                path = outputs / 'held' / f'{sc.HELD_STEMS[arm]}__s{seed}.npz'
                x = sc.load_held_output(path, arm, held_names, sha,
                                       expected_width=widths[arm][seed] if arm in widths else None)
                kind = 'k1' if arm == 'K1' else 'percept'
                record, _, _ = ct_diag.diagnose_vectors(x, labels['emotion'], kind, partitions)
                assert {'held', 'leiden_counts', 'graph'} <= set(record)
                diagnostics.append(dict(arm=arm, seed=seed, **record))
        save_json(root / 'smoke_diagnostics.json', dict(smoke=True, records=diagnostics))
        step('diagnostics')
    report.update(status='PASS', seconds=time.perf_counter() - started)
    save_json(root / 'smoke.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scratch', type=Path)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)
    run(args.scratch, args.device)


if __name__ == '__main__':
    main()
