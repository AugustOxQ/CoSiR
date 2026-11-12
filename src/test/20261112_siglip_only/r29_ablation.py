"""SC7: r28's four removals with K*'s caption input, held scoring and gate (vii).

Older code is imported from the main checkout through r29_paths, then patched
only for the duration of each call. Outputs use r14's weights/train/val/held
layout under OUT/ablation/siglip1/<K* source> (arms are removal names).
"""
import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P
P.use_older_rounds()
import r29_trainer as T
import r28_trainer as R
import r28_ablation as F
import r29_ladder as L
P.check_older_imports()

B = T.B
REMOVALS = R.REMOVALS
AMI_DRAWS, MAP_DRAWS = 2000, 10000      # spec §3.4: AMI 2,000 draws, mAP 10,000
BASE_HASHES = T.code_hashes


def code_hashes():
    return dict(BASE_HASHES(), **{Path(m.__file__).name: T.file_sha(m.__file__) for m in (R, R.L)},
                **{'r29_ablation.py': T.file_sha(__file__)})


def require_pass(smoke=False, source=None):
    """D3. Exempt: a smoke with the Qwen stand-in source. It writes only under SMOKE_OUT, reads no candidate record
    (no K*, only K0's frozen weights) and reports no number, so it may run before stage 1 (ruling of build 4)."""
    if smoke and source == 'qwen':
        return
    path = P.RECORDS / 'pass_rule.json'
    assert path.is_file() and json.loads(path.read_text()).get('status') == 'pass', \
        f'ablation needs {path} with status "pass" (D3)'


def kstar():
    path = P.RECORDS / 'prefix_pick.json'
    assert path.is_file(), f'missing {path}'
    pick = json.loads(path.read_text())
    assert {'kstar', 'kstar_form', 'p1_form', 'flag'} <= pick.keys(), f'{path}: missing shared interface keys'
    assert pick['kstar'] in ('siglip1', 'siglip1_prefix'), pick
    assert pick['kstar_form'] == {'siglip1': 'plain', 'siglip1_prefix': 'prefixed'}[pick['kstar']], pick
    return pick['kstar']


@contextmanager
def training_patches(row, source):
    """r28's removal configuration/loss; ticket 03 swaps only the caption input.

    Caption patches normally alias the arm to the source. Here each removal is
    its own arm, including sidecars and config.rg.arm, as r28's trainer expects.
    """
    assert row in REMOVALS and source in ('qwen', 'siglip1', 'siglip1_prefix')
    with patch.object(R, 'OUTPUTS', P.OUT), R.trainer_patches(row), \
            patch.object(T, 'code_hashes', code_hashes), \
            T.caption_patches(source, P.CAPTION_SOURCES[source][1]), \
            patch.object(B, 'row_arm', lambda _: row):
        yield


def train(row, seeds, device, root, *, smoke=False, source=None, max_epochs=None):
    require_pass(smoke, source)
    # qwen is solely the K0 stand-in for a smoke / fixed-batch regression.
    assert source is None or (smoke and source == 'qwen'), 'full ablations must use K*'
    source = source or kstar()
    root = T.check_root(root, smoke)
    seeds = list(seeds)
    assert row in REMOVALS and seeds and len(set(seeds)) == len(seeds) and set(seeds) <= set(P.SEEDS)
    assert max_epochs is None or (smoke and 1 <= max_epochs <= 2), 'only smoke may cap epochs (1 or 2)'
    for seed in seeds:
        path = B.record_path(root, row, seed)
        if path.exists():
            assert json.loads(path.read_text()).get('r29_code_sha256') == code_hashes(), \
                f'{path}: ablation training code changed'
    result = {}
    for seed in seeds:
        journal_path = root / 'train' / row / f'attempts__s{seed}.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else dict(
            row=row, seed=seed, source=source, r29_code_sha256=code_hashes(), attempts=[])
        assert (journal['row'], journal['seed'], journal['source'], journal['r29_code_sha256']) == (
            row, seed, source, code_hashes()), f'{journal_path}: retry requires the same seed, source and code'
        assert journal.get('status') != 'inconclusive', f'{row} s{seed}: stage inconclusive after two crashes (D21)'
        if journal.get('status') == 'ok':
            assert B.record_path(root, row, seed).is_file(), 'completed journal without a run record'
            with training_patches(row, source):
                result.update(B.train('siglip1', row, [seed], device, root, keep='all', smoke=smoke,
                                      max_epochs=max_epochs, graph_out=root))
            continue
        # A journal left running by process death consumes a try, too. Existing
        # artifacts are kept for inspection, including fully written weights
        # without a run record (B.train only moves incomplete weights aside).
        while len(journal['attempts']) < 2:
            if journal['attempts']:
                B.rb_student._move_aside(root, row, seed, B.rg_full.RECIPE)
                record_path = B.record_path(root, row, seed)
                if record_path.exists():
                    backup = root / 'replaced' / row / f'run__s{seed}__before_retry2.json'
                    assert not backup.exists(), f'will not overwrite {backup}'
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    record_path.rename(backup)
            attempt = dict(status='running')
            journal['attempts'].append(attempt)
            write_json(journal_path, journal)
            try:
                with training_patches(row, source):
                    run = B.train('siglip1', row, [seed], device, root, keep='all', smoke=smoke,
                                  max_epochs=max_epochs, graph_out=root)[seed]
                assert run.record['status'] == 'ok', run.record
            except Exception as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                write_json(journal_path, journal)
                continue
            attempt['status'] = journal['status'] = 'ok'
            write_json(journal_path, journal)
            result[seed] = run
            break
        else:
            journal['status'] = 'inconclusive'
            write_json(journal_path, journal)
            result[seed] = B.Run(dict(status='inconclusive', cause='two crashes (D21)', journal=str(journal_path)), None, None)
            break                   # no more seeds for this inconclusive arm
    return result


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')


def export(row, seeds, root, *, kinds=('held',), device=None, smoke=False, source=None):
    require_pass(smoke, source)
    assert row in REMOVALS
    assert source is None or (smoke and source == 'qwen'), 'full ablations must use K*'
    source = source or kstar()
    root = T.check_root(root, smoke)
    # T.export calls the ref context to obtain encoder dimensions. Route that
    # call to the removal so rg.arm, switches and graph names match its weights.
    def context(enc, _row, **kwargs):
        return R.training_context(enc, row, **kwargs)
    with patch.object(R, 'OUTPUTS', P.OUT), R.trainer_patches(row), \
            patch.object(B, 'training_context', context), patch.object(T, 'code_hashes', code_hashes):
        return T.export('siglip1', source, seeds, root, kinds=kinds, device=device, smoke=smoke, arm=row)


def shared_ami_draws(n_rows, n_cand_seeds, comp_seeded, n_draws, rng_seed=0):
    """r17 draws rows then seeds. Its seeded comparator normally draws APART;
    SC7 pairs a removal and K* by training seed, so both read the SAME draw.
    Delegating the transcript to the unseeded case consumes no extra seed RNG.
    """
    assert comp_seeded, 'SC7 compares two five-seed arms'
    for rows, seeds, _ in BASE_AMI_DRAWS(n_rows, n_cand_seeds, False, n_draws, rng_seed):
        yield rows, seeds, seeds


BASE_AMI_DRAWS = F.ns_clu.draws_iter


def paired_change(removal, reference, labels, k, *, ami_draws=AMI_DRAWS, map_draws=MAP_DRAWS):
    """Removal minus K*: paired rows/queries and training seeds, percentile 95%.
    No WIN/LEVEL label: SC7 is attribution, with no validation LEVEL unit.
    """
    F.r20_ret.same_queries(removal['ret'], reference['ret'])
    with patch.object(F.ns_clu, 'draws_iter', shared_ami_draws):
        ami = F.ns_clu.paired_cell(removal['parts'], reference['parts'], labels, k,
                                   metrics=('ami',), draws=ami_draws, rng_seed=0)['ami']
    mp = F.r20_ret.paired(removal['ret'], reference['ret'], draws=map_draws)
    return dict(ami={key: ami[key] for key in ('d', 'lo', 'hi', 'n_draws')}, map=dict(mp, n_draws=int(map_draws)))


def load_rows(root, arm, aspect, seed, list_sha, hashes, *, source, n=F.N_HELD, width=P.OUTPUT_WIDTH):
    """r28's checked float16 loader; K* and removal sidecars also bind source,
    caption width and the current store hash. `source=None` is frozen K0 only.
    """
    root = Path(root)
    path = root / 'held' / arm / f'{aspect}__s{seed}.npz'
    if source is None:
        assert root.resolve() == P.K0_ROOT.resolve() and arm == P.K0_ARM, 'only K0 has legacy sidecars'
    else:
        side = json.loads(path.with_suffix('.json').read_text())
        assert (side['cap_source'], side['cap_width'], side['cap_store_sha256'], side['encoder']) == (
            source, P.CAPTION_SOURCES[source][1], T.C.store_hash(source), B.backbone('siglip1')), path
    with patch.object(F, 'export_path', lambda *a, **kw: path), \
            patch.object(F, 'accepted_sidecar_arms', lambda _: {arm}):
        return F.load_rows(arm, aspect, seed, list_sha, hashes, n=n, width=width)


def score_aspect(root, arm, aspect, labels, names, list_sha, hashes, *, source):
    """Keep r28's point estimates, retaining partitions/AP for paired intervals.
    AMI clusters float16 exports cast to float32 WITHOUT re-normalising; mAP
    uses unit float32, aspect head, qe_k=0. All rows cluster before label masks.
    """
    k = F.CLASS_COUNT[aspect]
    assert F.ns_metrics.class_count(labels) == k
    parts, scored = [], []
    for seed in P.SEEDS:
        x = load_rows(root, arm, aspect, seed, list_sha, hashes, source=source)
        parts.append(F.ns_metrics.partitions(x, k, F.SET))
        scored.append(F.r20_ret.score_rows(F.r20_ret.unit(x), labels, names, 8))
        F.r20_ret.same_queries(scored[0], scored[-1])
    parts = np.stack(parts)
    (row,) = F.ns_clu.metric_table(arm, aspect, parts, labels, k, metrics=('ami',))
    ret = dict(scored[0], ap=np.stack([r['ap'] for r in scored]))
    mp = ret['ap'].mean(axis=1).tolist()
    cell = dict(k=k, seeds=list(P.SEEDS), ami_per_seed=row['per_seed'], ami_mean=row['mean'],
                map_per_seed=mp, map_mean=F.seed_mean(mp), n_labelled=int((labels >= 0).sum()),
                n_queries=len(ret['query_ids']), dropped=ret['dropped'])
    return dict(cell=cell, parts=parts, ret=ret)


def compare_replay(cells, list_sha, frozen):
    """D23: both means AND all five seed values within 1e-9, exact identities."""
    assert frozen['kind'] == 'r28_ablation_ref_replay'
    assert (frozen['encoder'], frozen['split'], frozen['arm'], frozen['seeds']) == (
        'siglip1', 'held', P.K0_ARM, list(P.SEEDS))
    checks = dict(list_sha256=list_sha == frozen['list_sha256'])
    for a in P.ASPECTS:
        cell = cells[a]
        checks[f'{a}/identity'] = (cell['seeds'] == frozen['seeds'] and cell['k'] == frozen['class_count'][a]
                                  and cell['n_queries'] == frozen['map_plain_cosine'][a]['n_queries']
                                  and cell['dropped'] == frozen['map_plain_cosine'][a]['dropped'])
        for metric, field in (('ami', 'ami'), ('map', 'map_plain_cosine')):
            actual = np.array([*cell[f'{metric}_per_seed'], cell[f'{metric}_mean']], dtype=np.float64)
            expected = np.array([*frozen[field][a]['per_seed'], frozen[field][a]['mean']], dtype=np.float64)
            assert actual.shape == expected.shape == (len(P.SEEDS) + 1,)
            checks[f'{a}/{metric}'] = bool(np.isfinite(actual).all() and np.max(np.abs(actual - expected)) <= 1e-9)
    return dict(checks=checks, tolerance=1e-9, **{'pass': all(checks.values())})


def require_committed(name):
    P.require_committed(P.RECORDS / name, 'SC7 reads held rows only after its inputs are committed (C13)')


def require_gate():
    """The one gate SC7 needs is vii_ablation (ticket 08 parks SC6 on the photo parts). Ladder rule (ticket 08): absent
    raises "not run"; fail before try 3 raises "still open"; fail at try 3 or parked returns 'parked' (SC7 inconclusive);
    pass returns 'pass' once the code and trainer-test hashes match."""
    status = L.gate_status(P.GATES.parent, 'vii_ablation')
    if status != 'pass':
        return 'parked'
    path = P.GATES / 'vii_ablation.json'
    gate = json.loads(path.read_text())
    assert gate['details']['code_sha256'] == code_hashes(), f'{path}: ablation code changed; rerun gate-vii'
    assert gate['details']['trainer_test_sha256'] == T.file_sha(HERE / 'test_r29_ablation.py'), 'trainer test changed'
    return 'pass'


def default_root(source, smoke=False):
    return (P.SMOKE_OUT if smoke else P.OUT) / 'ablation' / 'siglip1' / source


def score(root=None):
    """Full SC7 table, five seeds and all four removals; no held-driven picks.
    This readout is fixed by §3.4, so there is no validation readout selection
    or LEVEL range for SC7. SC1/SC2 records must already be committed.
    """
    require_pass()
    source = kstar()
    for name in ('prefix_pick.json', 'pass_rule.json'):
        require_committed(name)
    if require_gate() != 'pass':
        document = dict(kind='r29_ablation', claim='SC7', complete=False, status='inconclusive', set='artelingo',
                        split='held', encoder='siglip1', kstar=source, cause='gate vii_ablation is parked (a fail at try 3 parks)',
                        code_sha256=code_hashes())
        write_json(P.RECORDS / 'ablation.json', document)
        return document
    root = T.check_root(root or default_root(source))
    # Check crash status before reading ANY held rows, including K*'s.
    for row in REMOVALS:
        for seed in P.SEEDS:
            path = root / 'train' / row / f'attempts__s{seed}.json'
            if path.exists():
                assert json.loads(path.read_text())['status'] == 'ok', f'{row}: stage inconclusive (D21): {path}'
    names, list_sha, labels = F.t5.bind_labels()
    hashes, arms, deltas, cells = {}, {a: {} for a in (source, *REMOVALS)}, {r: {} for r in REMOVALS}, []
    for aspect in P.ASPECTS:
        reference = score_aspect(P.OUT / 'cstar/siglip1', source, aspect, labels[aspect], names, list_sha, hashes,
                                 source=source)
        arms[source][aspect] = reference['cell']
        for row in REMOVALS:
            removal = score_aspect(root, row, aspect, labels[aspect], names, list_sha, hashes, source=source)
            arms[row][aspect] = removal['cell']
            paired = paired_change(removal, reference, labels[aspect], F.CLASS_COUNT[aspect])
            deltas[row][aspect] = {}
            for measure in ('ami', 'map'):
                cand, comp = (item['cell'][f'{measure}_mean'] for item in (removal, reference))
                interval = paired[measure]
                assert abs(interval['d'] - (cand - comp)) <= 1e-12, 'point differs from paired scorer'
                per_seed = (np.array(removal['cell'][f'{measure}_per_seed']) -
                            np.array(reference['cell'][f'{measure}_per_seed'])).tolist()
                deltas[row][aspect][measure] = dict(interval, delta=interval['d'], per_seed=per_seed)
                k = F.CLASS_COUNT[aspect] if measure == 'ami' else None
                # id and field names as ticket 06's cells: `<set>/<encoder>/<arm>/<aspect>/<measure><k>/<comparator>`
                cells.append(dict(id=f'artelingo/siglip1/{row}/{aspect}/{measure}{"" if k is None else k}/{source}',
                                  set='artelingo', encoder='siglip1', arm=row, aspect=aspect, measure=measure, metric=measure,
                                  n_rows=int(len(labels[aspect])), k=k,
                                  comparator=source, cand=cand, comp=comp, value=cand, comparator_value=comp,
                                  **interval, r=None, outcome=None,
                                  seeds=list(P.SEEDS), per_seed=per_seed))
    drawn = {m: {c['n_draws'] for c in cells if c['measure'] == m} for m in ('ami', 'map')}
    assert all(len(v) == 1 for v in drawn.values()), drawn
    document = dict(kind='r29_ablation', claim='SC7', complete=True, status='ok', set='artelingo', split='held', encoder='siglip1',
                    kstar=source, kstar_form={'siglip1': 'plain', 'siglip1_prefix': 'prefixed'}[source],
                    cap_source=source, cap_width=P.CAPTION_SOURCES[source][1], seeds=list(P.SEEDS),
                    roots=dict(kstar=str(P.OUT / 'cstar/siglip1'), removals=str(root)),
                    list_sha256=list_sha, arms=arms, deltas_vs_kstar=deltas, cells=cells,
                    protocol=dict(readout='aspect head, plain cosine, qe_k=0', ami_draws=drawn['ami'].pop(), map_draws=drawn['map'].pop(),
                                  interval='paired percentile 95%', seed_coupling='shared', rng_seed=0),
                    code_sha256=code_hashes(), input_sha256=hashes)
    write_json(P.RECORDS / 'ablation.json', document)
    return document


def gate_vii():
    """Run the four fixed-batch tests, then replay all 15 K0 aspect/seed
    exports through THIS scorer. No candidate/pass record is needed in stage 0.
    K0's per-aspect timings in r28's full table sum to ~263 CPU seconds before
    input IO and trainer tests; use a queued CPU job for this CLI.
    """
    path = P.GATES / 'vii_ablation.json'
    prior = json.loads(path.read_text())['tries'] if path.exists() else 0
    assert prior < 3, f'{path}: three tries exhausted; ablation parked'
    command = [sys.executable, '-m', 'pytest', '-q', str(HERE / 'test_r29_ablation.py'),
               '-k', 'qwen_equals_r28_state_loss_and_gradients_on_fixed_batch']
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run(command, capture_output=True, text=True, cwd=P.REPO, env=env, timeout=240)
    details = dict(code_sha256=code_hashes(), trainer_test_sha256=T.file_sha(HERE / 'test_r29_ablation.py'),
                   trainer=dict(command=command, **{'pass': run.returncode == 0},
                                returncode=run.returncode, output=(run.stdout + run.stderr)[-6000:]),
                   replay=None)
    passed = False
    if run.returncode == 0:
        try:
            frozen_path = P.R28_RECORDS / 'ablation_ref_replay.json'
            frozen = json.loads(frozen_path.read_text())
            names, list_sha, labels = F.t5.bind_labels()
            hashes, cells = {}, {}
            for aspect in P.ASPECTS:
                cells[aspect] = score_aspect(P.K0_ROOT, P.K0_ARM, aspect, labels[aspect], names, list_sha,
                                             hashes, source=None)['cell']
                print(f'K0 replay {aspect}: AMI {cells[aspect]["ami_mean"]:.10f}, '
                      f'mAP {cells[aspect]["map_mean"]:.10f}', flush=True)
            comparison = compare_replay(cells, list_sha, frozen)
            details['replay'] = dict(comparison, source=str(frozen_path), source_sha256=T.file_sha(frozen_path),
                                     cells=cells, list_sha256=list_sha, input_sha256=hashes)
            passed = comparison['pass']
        except Exception as exc:
            details['replay'] = dict(error=f'{type(exc).__name__}: {exc}', **{'pass': False})
    status = 'pass' if passed else ('parked' if prior == 2 else 'fail')
    return T.C.write_gate_record(P.GATES, dict(status=status, details=details), gate='vii_ablation')


def smoke_checkpoint(root, device):
    """Exercise a real K0 checkpoint as K*'s stand-in while its pick is pending.
    The removal itself trains from scratch, exactly as r28; this is only a
    baseline loader/inference check, with no scored claim or fabricated pick.
    """
    root = T.check_root(root, smoke=True)
    aspect, seed = P.ASPECTS[0], P.SEEDS[0]
    weights = P.K0_ROOT / 'weights' / P.K0_ARM / f'{aspect}__s{seed}.pt'
    names, _, inputs = T.held_inputs(B.backbone('siglip1'))
    arrays = T.infer_sides(weights, dict(img=inputs['img'][:8]), device)
    assert arrays['img'].shape == (8, P.OUTPUT_WIDTH) and np.isfinite(arrays['img']).all()
    path = root / 'k0_standin' / f'{aspect}__s{seed}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    write_json(path.with_suffix('.json'), dict(smoke=True, weights=str(weights), weights_sha256=T.file_sha(weights),
                                              painting_names=list(names[:8]), note='K0 checkpoint stand-in; unscored'))


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'export'):
        cmd = commands.add_parser(command)
        cmd.add_argument('--row', choices=REMOVALS, required=True)
        cmd.add_argument('--seeds', nargs='+', type=int, choices=P.SEEDS, default=list(P.SEEDS))
        cmd.add_argument('--root', type=Path)
        cmd.add_argument('--device', default='cuda' if command == 'train' else None)
        cmd.add_argument('--smoke', action='store_true')
        cmd.add_argument('--k0-standin', action='store_true', help='smoke only: qwen input, K0 checkpoint baseline')
        if command == 'train':
            cmd.add_argument('--max-epochs', type=int)
        else:
            cmd.add_argument('--kind', nargs='+', choices=('val', 'held'), default=['held'])
    commands.add_parser('gate-vii', help='fixed-batch trainer tests + full frozen K0 scorer replay (queued CPU)')
    cmd = commands.add_parser('score', help='full five-seed SC7 table -> records/ablation.json')
    cmd.add_argument('--root', type=Path, help='removal root; default OUT/ablation/siglip1/<K*>')
    args = parser.parse_args(argv)
    if args.command == 'gate-vii':
        gate = gate_vii()
        print(f'vii_ablation: {gate["status"]}', flush=True)
        return 0 if gate['status'] == 'pass' else 2
    if args.command == 'score':
        doc = score(args.root)
        print(f'SC7: {len(doc["cells"])} cells -> {P.RECORDS / "ablation.json"}', flush=True)
        return 0
    assert not args.k0_standin or args.smoke, '--k0-standin is smoke only'
    source = 'qwen' if args.k0_standin else None
    require_pass(args.smoke, source)
    root = args.root or default_root(source or kstar(), args.smoke)
    if args.command == 'export':
        export(args.row, args.seeds, root, kinds=tuple(args.kind), device=args.device,
               smoke=args.smoke, source=source)
        return 0
    runs = train(args.row, args.seeds, args.device, root, smoke=args.smoke, source=source, max_epochs=args.max_epochs)
    for seed, run in runs.items():
        print(f'R29 ablation {args.row} s{seed}: {run.record["status"]}', flush=True)
    if any(run.record['status'] != 'ok' for run in runs.values()):
        return 2
    if args.k0_standin:
        export(args.row, args.seeds, root, smoke=True, source='qwen', device=args.device)
        smoke_checkpoint(root, args.device)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
