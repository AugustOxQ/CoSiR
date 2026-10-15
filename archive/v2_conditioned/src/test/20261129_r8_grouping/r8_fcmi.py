"""FCMI builder, DECISION_RULE M7 / section 5 (A:D2, D20, D21, SC2)."""
import r8_common as C

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from r8_fcmi_net import NetFCN, inference, LimitKmeans, CodeTest, UnDeterministic, DrawMax


class D21:
    """Four independent M7 tests; the step ledger is flushed before backward."""
    def __init__(self, loss_file=None):
        self.losses = []
        self.loss_file = loss_file

    def step(self, loss, epoch, batch):
        value = float(loss.detach().cpu())
        finite = bool(np.isfinite(value))
        entry = dict(epoch=epoch, batch=batch, total_loss=value if finite else None, finite=finite)
        self.losses.append(entry)
        if self.loss_file is not None:
            self.loss_file.write(json.dumps(entry, allow_nan=False) + '\n')
            self.loss_file.flush()
        if not finite:
            raise FloatingPointError(f'non-finite total loss at epoch {epoch}, batch {batch}')

    def record(self, exit_code, labels, wall_seconds):
        k = len(np.unique(labels)) if labels is not None else 0
        tests = dict(process=exit_code == 0, finite_loss=all(e['finite'] for e in self.losses),
                     clusters=k >= 2, wall_time=wall_seconds <= 3600)
        return {'pass': all(tests.values()), 'tests': tests, 'exit_code': exit_code,
                'wall_seconds': wall_seconds, 'clusters_before_merge': k, 'steps': len(self.losses)}


class BatchedTensorDataset(TensorDataset):
    """TensorDataset that slices a whole batch at once (same values as the default per-row collate)."""
    def __getitems__(self, idx):
        index = torch.as_tensor(idx, dtype=torch.long)
        return tuple(t[index] for t in self.tensors)


def identity_collate(batch):
    # __getitems__ already built the batch; a list matches the default collate's type.
    return list(batch)


def painting_loaders(x=None, context=None, batch_size=512, smoke=False):
    """Consume the common join verbatim; test/k-means sees only M1 scorer-train paintings."""
    context = C.ctx() if context is None else context
    x = C.x_paintings() if x is None else x
    assert x.shape == (context.n_paintings, 512) and x.dtype == np.float32
    assert np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    c = context.content
    assert c.shape == (len(x),) and c.dtype == np.int64 and c.min() >= 0 and c.max() < 64
    n = min(2048, len(x)) if smoke else len(x)
    dataset = BatchedTensorDataset(torch.from_numpy(x[:n].copy()), torch.from_numpy(c[:n].copy()),
                            torch.zeros(n, dtype=torch.long), torch.arange(n))
    # Controller default (fix round): M7 is silent here; follow upstream
    # DataSetMaster/dataset.py:424–427, including drop_last and the 100x test batch.
    # Batched slicing (BatchedTensorDataset + identity_collate) gives byte-identical batches and saves
    # about 0.6 s per epoch (the loop is CPU-bound); it changes no computation.
    train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(0), collate_fn=identity_collate)
    test = DataLoader(dataset, batch_size=batch_size * 100, shuffle=False, collate_fn=identity_collate)
    return train, test


def default_args(weight, device='cuda', smoke=False, output_dir=None, time_epochs=None):
    """main.py lines 20–82 defaults; M7 and explicit controller smoke/timing overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='Office')  # MNISTUSPS ReverseMNIST Office HAR MTFL
    parser.add_argument("--MnistTrain", default=1, type=int)
    parser.add_argument("--batch_size", help="batch size", default=512, type=int)
    parser.add_argument("--train_epoch", help="training epochs", default=300, type=int)

    parser.add_argument('--NetVersion', default='Shuffle', type=str)  # New, Old, Explicit, Shuffle
    parser.add_argument('--GroupWiseEncoder', action='store_true')
    parser.add_argument('--GroupWiseDecoder', default=1, type=int)
    parser.add_argument('--WithFeatureB', action='store_true')
    parser.add_argument('--FeatureType',
                        default='GlT_GaussainlizeAndTanh')  # Default Sigmoid Tanh Normalize Gaussainlize GlS_GaussainlizeAndSigmoid GlT_GaussainlizeAndTanh
    parser.add_argument('--BatchNormType', default='1001')  # 0000 -> 1111 1:Norm
    parser.add_argument('--RepresentationType', default='Normalize')  # Normalize None Relu
    parser.add_argument('--ActivationType',
                        default='Tanh')  # None Sigmoid Tanh Normalize Gaussainlize GlS_GaussainlizeAndSigmoid GlT_GaussainlizeAndTanh
    parser.add_argument("--representation_dim", help="", default=0, type=int)

    parser.add_argument('--RunInManuel', action='store_true')
    parser.add_argument("--VisualFreq", help="", default=20, type=int)
    parser.add_argument("--DrawTSNE", action='store_true')
    parser.add_argument("--DrawUmap", action='store_true')

    parser.add_argument("--WarmAll", default=20, type=int)
    parser.add_argument("--WarmBalance", default=0, type=int)
    parser.add_argument("--WarmConsistency", default=0, type=int)
    parser.add_argument("--WarmUpProto", default=0, type=int)
    parser.add_argument("--WarmOneHot", default=0, type=int)

    parser.add_argument("--SoftAssignmentTemperatureBalance", help="", default=0.10, type=float)
    parser.add_argument("--SoftAssignmentTemperatureHot", help="", default=0.10, type=float)
    parser.add_argument("--SoftAssignmentTemperatureSelfCons", help="", default=0.20, type=float)
    parser.add_argument("--Reconstruction", help="", default=1.0, type=float)
    parser.add_argument("--ReconstructionEpoch", default=999, type=int)
    parser.add_argument("--GlobalBalanceLoss", help="", default=0.0, type=float)
    parser.add_argument("--InfoGlobalLoss", help="", default=0.0, type=float)
    parser.add_argument("--InfoBalanceLoss", help="", default=0.04, type=float)
    parser.add_argument("--InfoFairLoss", help="", default=0.20, type=float)
    # parser.add_argument("--IndependentBalanceLoss", help="", default=0.0, type=float)
    parser.add_argument("--GroupWiseBalanceLoss", help="", default=0.0, type=float)  # row
    parser.add_argument("--ClusterWiseBalanceLoss", help="", default=0.0, type=float)  # line
    parser.add_argument("--BalanceLossType", default='KL')  # KL MSE MAE
    parser.add_argument("--BalanceLossNoDetach", action='store_true')
    parser.add_argument("--OneHot", help="", default=0.04, type=float)

    parser.add_argument("--loss_cons", help="", default=0.0, type=float)
    parser.add_argument("--loss_self_cons", help="", default=0.0, type=float)
    parser.add_argument("--SelfConsLossType", default='Assignment')  # Feature Assignment LogAssignment
    parser.add_argument("--Discriminative", help="", default=0.0, type=float)
    parser.add_argument("--DiscriminativeTest", help="", default=0, type=float)
    parser.add_argument("--reconstruct_all", help="", default=0, type=float)
    parser.add_argument("--Decenc", help="", default=0.0, type=float)

    parser.add_argument("--LearnRate", help="", default=1e-3, type=float)
    parser.add_argument("--LearnRateDecayType", default='None')  # Exp, Cosine
    parser.add_argument("--WeightDecay", default=0, type=float)
    parser.add_argument("--LearnRateWarm", default=0, type=int)
    parser.add_argument("--betas_a", help="", default=0.9, type=float)
    parser.add_argument("--betas_v", help="", default=0.999, type=float)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--resume", default='')

    parser.add_argument("--LambdaClu", help="alpha in our main paper Eq. 12", default=0.04, type=float)
    parser.add_argument("--LambdaFair", help="beta in our main paper Eq. 12", default=0.20, type=float)

    args = parser.parse_args([])
    assert weight in (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
    assert device in ('cpu', 'cuda')
    args.LambdaFair = weight
    args.InfoBalanceLoss = args.LambdaClu
    args.OneHot = args.LambdaClu
    args.InfoFairLoss = args.LambdaFair
    args.device = device
    args.train_epoch = 2 if smoke else 300
    assert not (smoke and time_epochs is not None)
    if time_epochs is not None:
        assert isinstance(time_epochs, int) and time_epochs > 0
        args.train_epoch = time_epochs
        args.WarmAll = 1  # controller timing mode: post-warm epochs exercise fairness
    if smoke:
        # Controller smoke-only overrides: epoch 2 must exercise clustering/fairness.
        args.WarmAll = 1
        args.VisualFreq = 2
    args.output_dir = str(output_dir) if output_dir is not None else None
    args.time_epochs = time_epochs
    return args


def native_labels(net, loader):
    """Final centre update is done by the caller, before this final argmax pass."""
    net.eval()
    labels = []
    with torch.no_grad():
        for x, g, _, idx in loader:
            assert torch.equal(idx, torch.arange(len(labels), len(labels) + len(x)))
            h = net.encode(x.to(net.args.device), g.to(net.args.device))
            c = net.encode_class(h)
            assert torch.isfinite(c).all()
            labels.extend(torch.argmax(c, dim=1).cpu().tolist())
    return np.unique(np.asarray(labels, dtype=np.int64), return_inverse=True)[1].astype(np.int64)


def image_space(net, x):
    """M7's 17-value shared representation; gallery images need no protected group."""
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 512
    assert np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            batch = torch.from_numpy(x[start:start + 512].copy()).to(net.args.device)
            h = net.encoder_linear[0](net.encoder[0](batch))
            parts.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy())
    phi = np.concatenate(parts).astype(np.float32)
    assert phi.shape == (len(x), 17) and np.isfinite(phi).all()
    assert np.allclose(np.linalg.norm(phi, axis=1), 1, atol=1e-6)
    return phi


WEIGHTS = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
SOURCE = '/data/SSD2/third_party/FCMI/'
CHECKPOINT_ROOT = Path('/data/SSD2/cosir_r8/fcmi')


# R7 r7_common.sha_file, copied to stay within 00-index's r8_common interface.
def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def run_paths(weight, smoke, time_epochs=None):
    name = f'FC{WEIGHTS.index(weight) + 1}'
    trial = smoke or time_epochs is not None
    root = C.SMOKE / 'fcmi' / name if trial else C.RESULTS
    if time_epochs is not None:
        root = C.SMOKE / 'fcmi' / f'time_epochs{time_epochs}' / name
    directory = root if trial else root / 'fcmi' / name
    # Section 5: FC3's loss ledger and pre-merge labels sit beside the stage-1 record.
    prefix = root / 'stage1_fcmi' if weight == 0.2 else directory / 'd21'
    # Fix-round controller instruction: large smoke checkpoints stay off /project.
    # The output-root hash separates CPU/GPU smokes and timing runs without overwriting.
    run = name + '-' + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    checkpoint = CHECKPOINT_ROOT / 'smoke' / run if trial else CHECKPOINT_ROOT / name
    return dict(name=name, directory=directory, checkpoint=checkpoint,
                verdict=prefix.with_suffix('.json'), losses=prefix.with_name(prefix.name + '_losses.jsonl'),
                labels=prefix.with_name(prefix.name + '_labels.npz'),
                started=directory / 'started.json', labels_done=directory / 'labels_done.json',
                status=directory / 'worker.json', log=directory / 'worker.log',
                timing=directory / 'timing.json',
                space=root / 'spaces' / f'{name}.npz', meta=root / 'spaces' / f'{name}.json')


def open_once(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open('w' if Path(path).resolve().is_relative_to(C.SMOKE.resolve()) else 'x')


def configure(device):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'M7 CUBLAS_WORKSPACE_CONFIG'
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(name) == '8', f'{name} must be 8'
    torch.set_num_threads(8)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert device in ('cpu', 'cuda')
    if device == 'cuda':
        assert torch.cuda.is_available(), 'real runs require the controller GPU lock'


def worker(weight, device, smoke, time_epochs=None):
    """Child writes a loss ledger even if a later step or process fails."""
    paths = run_paths(weight, smoke, time_epochs)
    started = None
    labels_finished = None
    error = None
    try:
        C.assert_rule()
        configure(device)
        assert version('faiss-cpu') == '1.15.1', 'M7 faiss-cpu version'
        input_shas = C.assert_inputs(keys=(SOURCE + 'networkExplicitDecoupleShuffle.py',
                                          SOURCE + 'main.py',
                                          '/data/SSD2/pre_extract/artelingo/features/metadata.json'), r7=False)
        context = C.ctx()
        train, test = painting_loaders(smoke=smoke)
        assert context.n_paintings == 36518 and len(context.scorer_train) == 183694
        assert len(test.dataset) == (2048 if smoke else 36518)
        # Record all already-verified common inputs, without deriving any common join.
        # nonheld_images() also verifies the image boundary before model construction.
        if time_epochs is None:
            images = C.nonheld_images()
            assert images.dtype == np.int32 and images.shape == (49121,) and (np.diff(images) > 0).all()
            painting_images = C.painting_image()
        args = default_args(weight, device, smoke, paths['checkpoint'], time_epochs)
        with open_once(paths['losses']) as loss_file:
            guard = D21(loss_file)
            started = time.monotonic()
            C.write_json_once(paths['started'], dict(monotonic=started))
            net = NetFCN(input_dim=512, class_num=17, group_num=64, args=args).to(device)
            epoch_seconds = net.run(args.train_epoch, train, test, args, guard)
            if time_epochs is not None:
                C.write_json_once(paths['timing'], dict(smoke=True, time_epochs=time_epochs,
                    n_paintings=len(test.dataset), training_batches=len(train), test_batch_size=test.batch_size,
                    settings=vars(args), epoch_seconds=epoch_seconds, input_sha256=input_shas))
                return 0  # timing only: no final labels, image space, or stage evidence
            # Refit centres on the final encoder, then assign using those updated centres.
            _, _, _, _, centers = inference(net, test)
            net.update_cluster_center(centers)
            labels = native_labels(net, test)
            labels_finished = time.monotonic()
            assert labels.dtype == np.int64 and labels.shape == (len(test.dataset),)
            C.write_npz_once(paths['labels'], labels=labels,
                             painting_index=np.arange(len(labels), dtype=np.int64),
                             painting_ids=context.painting_ids[:len(labels)],
                             painting_image=painting_images[:len(labels)])
            C.write_json_once(paths['labels_done'], dict(monotonic=labels_finished,
                                                       epoch_seconds=epoch_seconds))
            if not smoke and not guard.record(0, labels, labels_finished - started)['pass']:
                return 0  # parent records the failed D21; no space is published
            # rule gap: M7 says "then merge_small", but 00-index assigns the native-label
            # merge and r8_groupings files to ticket 07. Deliver pre-merge labels + phi here;
            # ticket 07 applies the 200-row merge in this saved space.
            if smoke:
                image_index = np.sort(painting_images[:2048]).astype(np.int32)
                x = C.x_images()[np.searchsorted(images, image_index)]
            else:
                image_index, x = images, C.x_images()
            phi = image_space(net, x)
            C.write_npz_once(paths['space'], phi=phi, image_index=image_index)
            # Include the hashes verified by the common layer at these boundaries.
            input_shas.update(C.assert_inputs(keys=(
                '20261031_pseudo_partitions/results/partitions.npz',
                '20261117_reader_fix_csd/results/rb_halves.npz',
                '/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json',
                '/data/SSD2/pre_extract/artelingo/style_csd_vitl/row_to_image.npy',
                '/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy',
                '/data/SSD2/pre_extract/artelingo_clip224/paintings.json'), r7=False))
            C.write_json_once(paths['meta'], dict(builder='FCMI', smoke=smoke, settings=vars(args),
                mainlauncher=dict(LimitKmeans=LimitKmeans, CodeTest=CodeTest,
                                  UnDeterministic=UnDeterministic, DrawMax=DrawMax),
                dimensions=dict(input_dim=512, group_num=64, class_num=17, representation_dim=17),
                faiss=dict(version=version('faiss-cpu'), index='IndexFlatL2', niter=300, nredo=10,
                           seed='Clustering default', max_points_per_centroid=1000, min_points_per_centroid=10),
                deterministic_algorithms=True, allow_tf32=False, cublas_workspace_config=':4096:8',
                upstream_commit='9750ccfc91a64c10364977302fc53875ed39a23f',
                input_sha256=input_shas, native_labels=str(paths['labels']),
                native_labels_sha256=sha_file(paths['labels']),
                space_sha256=sha_file(paths['space']), representation='unit encoder_linear[0](encoder[0](x))',
                n_images=len(phi), n_paintings=len(labels), epoch_seconds=epoch_seconds))
    except (Exception, SystemExit) as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        C.write_json_once(paths['status'], dict(error=error, started=started,
            wall_seconds=(labels_finished or time.monotonic()) - started if started is not None else 0,
            labels_finished=labels_finished))
    return 0


def read_json(path, keys):
    record = json.loads(path.read_text())
    assert set(keys) <= record.keys(), f'{path}: missing keys'
    assert record['rule_sha256'] == sha_file(C.RULE), f'{path}: rule SHA mismatch'
    return record


def build(weight=0.2, device='cuda', smoke=False, time_epochs=None):
    """Controller entry point; external flock wait is outside the timed child."""
    assert weight in WEIGHTS
    trial = smoke or time_epochs is not None
    assert not (smoke and time_epochs is not None)
    assert time_epochs is None or (isinstance(time_epochs, int) and time_epochs > 0)
    assert device == 'cuda' or trial, 'CPU is for tests / smoke modes only'
    C.assert_rule()
    paths = run_paths(weight, smoke, time_epochs)
    if not trial and weight != 0.2:
        stage1 = read_json(C.RESULTS / 'stage1_fcmi.json', ('pass', 'tests', 'smoke', 'weight', 'device'))
        assert isinstance(stage1['pass'], bool), 'SC2 pass must be bool'
        assert stage1['pass'] and not stage1['smoke'] and stage1['weight'] == 0.2
        assert stage1['device'] == 'cuda' and all(stage1['tests'].values()), 'SC2 failed: all FC runs drop'
    # Require fresh smoke evidence too: stale child markers must never mask a failure.
    targets = [paths[k] for k in ('verdict', 'losses', 'labels', 'started', 'labels_done', 'status', 'log', 'space', 'meta', 'timing')]
    assert all(not p.exists() for p in targets), 'FCMI output exists (use a fresh smoke RESULTS root for another smoke)'
    assert not paths['checkpoint'].exists(), 'checkpoint run directory exists (write once)'
    paths['directory'].mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--weight', str(weight), '--device', device, '--worker']
    if smoke:
        command.append('--smoke')
    if time_epochs is not None:
        command.extend(['--time-epochs', str(time_epochs)])
    timed_out = False
    started = None
    printed_epochs = 0
    with open_once(paths['log']) as log:
        with paths['log'].open() as live_log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            pending = ''
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pending += live_log.read()
                    lines = pending.split('\n')
                    pending = lines.pop()
                    for line in lines:
                        if line.startswith('epoch_wall_seconds='):
                            printed_epochs += 1
                            print(f'epoch={printed_epochs} {line}', flush=True)
                    if paths['started'].exists() and started is None:
                        # A writer may still be serialising the tiny marker; retry next tick.
                        try:
                            started = read_json(paths['started'], ('monotonic',))['monotonic']
                        except json.JSONDecodeError:
                            continue
                    if started is not None and not paths['labels_done'].exists() and time.monotonic() - started > 3600:
                        timed_out = True
                        process.kill()
                        process.wait(timeout=30)
            exit_code = process.returncode
    guard = D21()
    if paths['losses'].exists():
        for line in paths['losses'].read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                assert exit_code != 0, 'partial loss ledger from successful child'
                break  # abrupt process death can leave the final line incomplete
            assert {'epoch', 'batch', 'total_loss', 'finite'} <= entry.keys()
            assert entry['finite'] == (entry['total_loss'] is not None and np.isfinite(entry['total_loss']))
            guard.losses.append(entry)
    if exit_code == 0:
        epochs, batches = (2, 4) if smoke else (time_epochs or 300, 71)
        assert len(guard.losses) == epochs * batches, 'loss ledger step count'
        assert [(e['epoch'], e['batch']) for e in guard.losses] == [
            (epoch, batch) for epoch in range(epochs) for batch in range(batches)], 'loss ledger step alignment'
    if time_epochs is not None:
        if paths['timing'].exists():
            timing = read_json(paths['timing'], ('epoch_seconds', 'n_paintings', 'settings'))
            assert timing['n_paintings'] == 36518 and len(timing['epoch_seconds']) == time_epochs
            for epoch, seconds in enumerate(timing['epoch_seconds'], 1):
                if epoch > printed_epochs:
                    print(f'epoch={epoch} epoch_wall_seconds={seconds:.6f}', flush=True)
        print(f"{paths['name']} timing_epochs={time_epochs} exit={exit_code}", flush=True)
        return exit_code if exit_code != 0 else (0 if all(e['finite'] for e in guard.losses) else 1)
    labels = None
    if paths['labels'].exists() and paths['labels_done'].exists() and not timed_out:
        with np.load(paths['labels'], allow_pickle=False) as z:
            assert {'labels', 'painting_index', 'painting_ids', 'painting_image'} <= set(z.files)
            labels = z['labels']
            n = 2048 if smoke else 36518
            context = C.ctx()
            assert labels.shape == (n,) and labels.dtype == np.int64
            assert np.array_equal(np.unique(labels), np.arange(len(np.unique(labels))))
            assert np.array_equal(z['painting_index'], np.arange(n))
            assert np.array_equal(z['painting_ids'], context.painting_ids[:n])
            assert np.array_equal(z['painting_image'], C.painting_image()[:n])
    status = read_json(paths['status'], ('wall_seconds', 'error', 'started')) if paths['status'].exists() else None
    if status is not None:
        wall_seconds = status['wall_seconds']
    else:
        if started is None and paths['started'].exists():
            started = read_json(paths['started'], ('monotonic',))['monotonic']
        wall_seconds = time.monotonic() - started if started is not None else 0
    if timed_out:
        wall_seconds = max(wall_seconds, 3600.000001)
    result = guard.record(exit_code, labels, wall_seconds)
    # A wiring smoke is evidence for the wiring only, and never an SC2 decision.
    wiring_pass = result['pass']
    result['pass'] = wiring_pass and not smoke
    if smoke:
        result['wiring_pass'] = wiring_pass
    result.update(smoke=smoke, weight=weight, device=device,
        settings=vars(default_args(weight, device, smoke, paths['checkpoint'])),
        training_batches=4 if smoke else 71, test_batch_size=51200,
        grouping=paths['name'], error='construction-to-labels exceeded 60 minutes' if timed_out else
        (status['error'] if status else 'worker ended without status'),
        loss_log=str(paths['losses']), labels_before_merge=str(paths['labels']) if labels is not None else None,
        loss_log_sha256=sha_file(paths['losses']) if paths['losses'].exists() else None,
        labels_sha256=sha_file(paths['labels']) if labels is not None else None,
        drops=list(f'FC{i}' for i in range(1, 7)) if not smoke and weight == 0.2 and not result['pass'] else [])
    C.write_json_once(paths['verdict'], result)
    print(f"{paths['name']} smoke={smoke} D21={result['pass']} exit={exit_code} wall_seconds={wall_seconds:.3f}", flush=True)
    if paths['labels_done'].exists():
        timing = read_json(paths['labels_done'], ('epoch_seconds',))
        for epoch, seconds in enumerate(timing['epoch_seconds'], 1):
            if epoch > printed_epochs:
                print(f'epoch={epoch} epoch_wall_seconds={seconds:.6f}', flush=True)
    return exit_code if exit_code != 0 else (0 if wiring_pass else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=float, required=True, choices=WEIGHTS)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    # rule gap: ticket 06 explicitly calls its --smoke CPU; M7 / the task also mention
    # a GPU wiring smoke. Default smoke to CPU and allow --device cuda for controller lock runs.
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true', help='2 epochs, first 2048 paintings, no SC2; default CPU')
    modes.add_argument('--time-epochs', type=int, metavar='N', help='N full-painting epochs, WarmAll=1, smoke outputs only; default CUDA')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    device = args.device or ('cpu' if args.smoke else 'cuda')
    assert args.smoke or args.time_epochs is not None or device == 'cuda', 'real runs require CUDA'
    run = worker if args.worker else build
    return run(args.weight, device, args.smoke, args.time_epochs)


if __name__ == '__main__':
    raise SystemExit(main())
