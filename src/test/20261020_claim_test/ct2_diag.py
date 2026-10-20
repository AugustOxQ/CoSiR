"""Stage 2 diagnostics over the full root; CPU only, at most eight threads.

Runs ct_diag.run unchanged (V, ARI, FMI with random baselines, k-means at
Leiden's count, Leiden 3a and 3b, in-sample reading, margins, target and
encoder rows, paper run table). Only inputs, sizes and paths change: the stage
2 verdict gate, the 42,969 fit paintings, ct2's weight checks and the output
file. The 3b fit graph and vote are exact kNN in blocks (ct_diag's code).
Diagnostics decide nothing and never write the verdict.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ct_diag
import ct_held
import ct_score as sc
import ct2_d2
import ct2_stage2a
import ct2_data
import ct2_percept
import ct2_verdict

HERE = Path(__file__).resolve().parent
FULL_ROOT = ct2_verdict.FULL_ROOT
FIT_ROWS = 42969
PATCH_BLOCK = 2048  # fit paintings per patch read: 2,048 x 50 x 512 float32 = 210 MB


def read_verdict(path):
    """The first file boundary: the stage 2 verdict must exist before any input is read."""
    path = Path(path)
    assert path.is_file(), 'diagnostics require verdict_stage2.json first'
    record = json.loads(path.read_text())
    assert record['stage'] == 2 and record['outcome_id'] in ('S2-GO', 'S2-MM', 'S2-IN', 'S2-NO')
    assert set(record['eligibility']['eligible_seeds']) == set(sc.ARM_ORDER)
    assert record['eligibility']['eligible_seeds']['K1'] == sc.SEEDS
    assert record['eligibility']['eligible_seeds']['B1'] == sc.SEEDS
    return record


class FitInference(ct_diag.FitInference):
    """CPU inference on the full train paintings with ct2's saved weights."""

    def __init__(self, output_root, held_names, mode='full', s=None):
        import ct_d2
        import ct_percept as pc
        import torch
        self.d2, self.pc, self.torch = ct_d2, pc, torch
        torch.set_num_threads(8)
        self.mode, self.root = mode, Path(output_root)
        s = ct2_data.split2(mode) if s is None else s
        self.sample_sha256 = pc.pc_data.fingerprint(s)
        self.names = sorted(s['all_painting_ids'][s['fit']].tolist())
        assert len(self.names) == len(set(self.names)) == (FIT_ROWS if mode == 'full' else 13400)
        assert not set(self.names) & set(held_names)
        self.labels = ct_held.painting_labels(self.names)
        self.inputs, self.sources = {}, {}

    def batches(self, kind, size):
        """Fit inputs in name order; patches are read per block, never all at once."""
        if kind == 'patch':
            for start in range(0, len(self.names), PATCH_BLOCK):
                yield ct_held.clip_patches(self.names[start:start + PATCH_BLOCK])
        else:
            yield self.features(kind)

    def predict(self, row, aspect, seed):
        torch, pc, d2 = self.torch, self.pc, self.d2
        if row == 'K1':
            meta = ct2_d2._weights_info(self.mode, self.root, aspect, seed)
            path = d2._weight_path(self.root, aspect, seed)
            _, views = d2.ms_train.build_model(d2.ROWS['D2'], list(d2.ASPECTS), torch.device('cpu'))
            view = views[aspect]
            view.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model, kind, width, dtype = view['img'], 'qwen_default', 2048, np.float16
        else:
            settings, cut, arm = ct_diag.COMPARATORS[row]
            if row == 'B1' and aspect == 'style':
                arm = 'P2'
            path = self.root / 'percept' / settings / 'stage2' / cut / f'{arm}__s{seed}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            c = meta['config']
            assert (c['settings'], c['cut'], c['arm'], c['seed']) == (settings, cut, arm, seed)
            assert not c['smoke'] and c['mode'] == self.mode and meta['sample_sha256'] == self.sample_sha256
            assert sc._sha_file(path.with_suffix('.pt')) == meta['weights_sha256']
            model = pc.build_mapper(c, meta['K'])
            state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
            model.load_state_dict(state['mapper'])
            kind, width, dtype = pc.ARMS[arm][1], meta['K'], np.float32
        model.eval()
        prob = np.empty((len(self.names), width), dtype=dtype)
        done = 0
        with torch.no_grad():
            for x in self.batches(kind, 256):
                if kind == 'qwen_default' and row != 'K1':
                    x = x / np.linalg.norm(x, axis=1, keepdims=True)
                for start in range(0, len(x), 256):
                    value = model(torch.from_numpy(np.ascontiguousarray(x[start:start + 256])))
                    if row != 'K1':
                        value = value.sigmoid()
                    prob[done:done + len(value)] = value.numpy().astype(dtype)
                    done += len(value)
        assert done == len(self.names) and np.isfinite(prob).all()
        if row == 'K1':
            assert np.all(np.abs(np.linalg.norm(prob.astype(np.float32), axis=1) - 1) < 1e-3)
        else:
            assert ((prob >= 0) & (prob <= 1)).all()
        self.sources[f'{row}/{aspect}/{seed}'] = dict(path=str(path.with_suffix('.pt')),
                                                     weights_sha256=meta['weights_sha256'],
                                                     training_device=meta['config']['device'],
                                                     inference_device='cpu')
        return prob.astype(np.float32)


def run(verdict_path=HERE / 'verdict_stage2.json', root=FULL_ROOT, destination=HERE / 'diag_stage2.json',
        *, workers=8, n_resamples=2000, mode='full'):
    """ct_diag.run over the stage 2 root, after the stage 2 verdict."""
    verdict_path, destination = Path(verdict_path), Path(destination)
    assert verdict_path.name == 'verdict_stage2.json', 'stage 2 verdict path required'
    assert destination.name == 'diag_stage2.json', 'stage 2 diagnostics path required'
    read_verdict(verdict_path)  # refuse before any other read or patch
    with patch.object(ct_diag, 'read_verdict', read_verdict), \
            patch.object(ct_diag, 'FitInference', lambda out, names: FitInference(out, names, mode)), \
            patch.object(ct_diag, 'HERE', destination.resolve().parent):
        report = ct_diag.run(verdict_path, root, destination, workers=workers, n_resamples=n_resamples)
    saved = json.loads(destination.read_text())
    assert saved == json.loads(json.dumps(report, allow_nan=False)), 'diagnostics fields changed after ct_diag.run'
    saved['ct2_code_sha256'] = {str(path): sc._sha_file(path) for path in
                                (Path(__file__), Path(ct2_d2.__file__),
                                 Path(ct2_percept.__file__), Path(ct2_data.__file__))}
    assert all(saved[key] == report[key] for key in ('verdict_sha256', 'held_sha256', 'eligibility'))
    assert sc._sha_file(verdict_path) == saved['verdict_sha256'], 'verdict changed during diagnostics'
    sc._atomic_json(destination, saved)
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verdict', type=Path, default=HERE / 'verdict_stage2.json')
    parser.add_argument('--root', type=Path, default=FULL_ROOT)
    parser.add_argument('--out', type=Path, default=HERE / 'diag_stage2.json')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--stage2a', type=Path, default=HERE / 'stage2a.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    ct2_stage2a.require_pass(args.stage2a, args.device)  # V1: no diagnostics without a Stage 2a pass
    run(args.verdict, args.root, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
