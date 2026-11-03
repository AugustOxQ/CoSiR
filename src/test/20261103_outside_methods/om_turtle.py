"""r18 ticket 02: our own TURTLE fit for one representation space, and gate (ii).

TURTLE (Gadetsky, Cai and Brbic, "Let Go of Your Labels with Unsupervised Transfer", ICML 2024, arXiv 2406.07236)
learns a linear task encoder whose softmax labelling of the images is the one a linear classifier, trained on those
labels, predicts best on the same images, minus a term that keeps the clusters balanced. The code here is written
from the paper's description and from the facts brief (`om_facts_brief.md` §B), and checked against the official
code's saved run (gate (ii)). Spec D12: no line of the official code is copied.

Fixed by the spec (`om_paths.TURTLE`): gamma 10, 6,000 outer steps, 10 inner steps, Adam 1e-3 for both, a batch of
10,000 rows (n when smaller), the inner classifier drawn afresh each outer step (cold start). K is given.

The random streams, in the official order: the torch seed; the task encoder's own init; one inner init before the
loop (its draw is kept in the stream although the first outer step draws a new one); then each outer step draws its
batch indices (numpy, `RandomState(seed).choice(n, min(10000, n), replace=False)`) and then the inner init (torch).
No label is read anywhere: the official every-20-steps accuracy evaluation draws no random number and is left out.

CLI: `python om_turtle.py gate-ii --device cpu|cuda` writes `OUT/stage0/gate_ii.json`.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_paths as P  # noqa: E402

OUT = P.OUT
GAMMA = P.TURTLE['gamma']
INNER_STEPS = P.TURTLE['inner_steps']
OUTER_LR = P.TURTLE['outer_lr']
INNER_LR = P.TURTLE['inner_lr']
BATCH = P.TURTLE['batch_size']
BETAS = (0.9, 0.999)


def _no_tf32():
    """TF32 off for the matrix products (a CUDA fit must follow the float32 arithmetic of the reference)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def _init_inner(d, K, device):
    """One fresh inner linear classifier: [weight [K, d], bias [K]] as leaf tensors on `device`, drawn on the CPU
    stream with torch's default Linear init (weight, then bias), then moved."""
    lin = torch.nn.Linear(d, K)
    return [lin.weight.detach().to(device).requires_grad_(), lin.bias.detach().to(device).requires_grad_()]


def _effective_weight(v, g):
    """Weight-normalised weight: each row of v rescaled to the norm in g."""
    return v * (g / v.norm(dim=1, keepdim=True))


def batch_indices(n, seed, draws):
    """The first `draws` batch index arrays the loop uses: the numpy stream after the official seed setting
    (`np.random.seed(seed)`), `choice(n, min(10000, n), replace=False)` each time. A list of int arrays."""
    rs = np.random.RandomState(seed)
    return [rs.choice(n, size=min(BATCH, n), replace=False) for _ in range(draws)]


def fit(z, K, seed, device, outer_steps=P.TURTLE['outer_steps'], threads=None):
    """TURTLE on one space. z float32 [n, d] (numpy or torch). Returns labels int32 [n] (argmax on every row),
    weight float32 [K, d] and bias float32 [K] (effective, the weight norm removed), loss and entropy (the last
    outer step's prediction error and entropy term), n_clusters (non-empty), seconds. Seeds torch's global stream
    (as the official code does); numpy's global stream is not touched."""
    t0 = time.time()
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    assert isinstance(z, np.ndarray) and z.dtype == np.float32 and z.ndim == 2, 'z must be float32 [n, d]'
    assert np.isfinite(z).all(), 'z has non-finite values'
    n, d = z.shape
    assert 2 <= K <= n and outer_steps >= 1
    device = torch.device(device)
    _no_tf32()
    old_threads = torch.get_num_threads()
    if threads is not None and device.type == 'cpu':
        torch.set_num_threads(threads)
    try:
        torch.manual_seed(seed)
        rs = np.random.RandomState(seed)
        zt = torch.from_numpy(z).to(device)                      # data stays on the device
        enc = torch.nn.Linear(d, K)                              # task encoder: init draw
        g = enc.weight.detach().norm(dim=1, keepdim=True).to(device).requires_grad_()
        v = enc.weight.detach().to(device).requires_grad_()
        b = enc.bias.detach().to(device).requires_grad_()
        opt = torch.optim.Adam([b, g, v], lr=OUTER_LR, betas=BETAS)
        _init_inner(d, K, device)                                # drawn before the loop, as in the official order
        batch = min(BATCH, n)
        loss = entropy = float('nan')
        for _ in range(outer_steps):
            idx = torch.from_numpy(rs.choice(n, size=batch, replace=False)).to(device)
            zb = zt[idx]
            p = F.softmax(F.linear(zb, _effective_weight(v, g), b), dim=1)       # [batch, K]
            iw, ib = _init_inner(d, K, device)                   # cold start
            inner = torch.optim.Adam([iw, ib], lr=INNER_LR, betas=BETAS)
            target = p.detach()
            for _ in range(INNER_STEPS):
                inner.zero_grad()
                F.cross_entropy(F.linear(zb, iw, ib), target).backward()
                inner.step()
            opt.zero_grad()
            pred_error = F.cross_entropy(F.linear(zb, iw, ib).detach(), p)
            ent = torch.special.entr(p.mean(0)).sum()
            (pred_error - GAMMA * ent).backward()
            opt.step()
            loss, entropy = float(pred_error.detach()), float(ent.detach())
        with torch.no_grad():
            w_eff = _effective_weight(v, g)
            labels = F.linear(zt, w_eff, b).argmax(dim=1).cpu().numpy().astype(np.int32)
            weight, bias = w_eff.cpu().numpy().astype(np.float32), b.detach().cpu().numpy().astype(np.float32)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    finally:
        torch.set_num_threads(old_threads)
    assert labels.shape == (n,) and weight.shape == (K, d) and bias.shape == (K,)
    return {'labels': labels, 'weight': weight, 'bias': bias, 'loss': loss, 'entropy': entropy,
            'n_clusters': int(len(np.unique(labels))), 'seconds': time.time() - t0}


# ---- gate (ii) -------------------------------------------------------------------------------------------------

def _gate_path():
    return Path(OUT) / 'stage0' / 'gate_ii.json'


def gate_ii(device):
    """Gate (ii) (D23). CPU part: our fit on the parity inputs equals the official run's saved weight and bias
    within 1e-5, the partition is identical, the batch draws equal numpy's. CUDA part: the GPU partition's AMI
    against our CPU one at least 0.99. Writes `OUT/stage0/gate_ii.json`: the parts run so far; `pass` is true only
    when both parts are present and pass. Returns the JSON content."""
    from sklearn.metrics import adjusted_mutual_info_score
    c = P.PARITY_TURTLE
    z = np.load(P.parity_file('inputs'))['qwen']
    ref = np.load(P.parity_file('turtle'))
    path = _gate_path()
    res = {}
    if path.exists():
        import json
        res = json.loads(path.read_text())
    cpu = fit(z, c['K'], c['seed'], 'cpu', outer_steps=c['outer_steps'], threads=c['threads'])
    if device == 'cpu':
        dw = float(np.abs(cpu['weight'] - ref['weight']).max())
        db = float(np.abs(cpu['bias'] - ref['bias']).max())
        same = bool(np.array_equal(cpu['labels'], ref['partition']))
        bc = P.BATCH_DRAW_CHECK
        batch_same = True
        for n in bc['n']:
            np.random.seed(bc['seed'])
            want = [np.random.choice(n, size=min(BATCH, n), replace=False) for _ in range(bc['draws'])]
            got = batch_indices(n, bc['seed'], bc['draws'])
            batch_same &= all(np.array_equal(a, w) for a, w in zip(got, want))
        res['cpu'] = {'pass': bool(dw < P.TOL_TURTLE and db < P.TOL_TURTLE and same and batch_same),
                      'max_abs_weight': dw, 'max_abs_bias': db, 'partition_identical': same,
                      'batch_indices_identical': bool(batch_same), 'tol': P.TOL_TURTLE, 'n_clusters': cpu['n_clusters'],
                      'threads': c['threads'], 'seconds': cpu['seconds'], 'K': c['K'], 'outer_steps': c['outer_steps'],
                      'seed': c['seed'], 'reference_sha256': P.PARITY_SHA256['turtle']}
    else:
        assert device == 'cuda' and torch.cuda.is_available()
        gpu = fit(z, c['K'], c['seed'], 'cuda', outer_steps=c['outer_steps'])
        ami = float(adjusted_mutual_info_score(cpu['labels'], gpu['labels']))
        res['cuda'] = {'pass': bool(ami >= P.GPU_CPU_AMI), 'ami_vs_cpu': ami, 'threshold': P.GPU_CPU_AMI,
                       'n_clusters': gpu['n_clusters'], 'seconds': gpu['seconds'], 'K': c['K'],
                       'outer_steps': c['outer_steps'], 'seed': c['seed'],
                       'tf32': bool(torch.backends.cuda.matmul.allow_tf32)}
    res['pass'] = bool('cpu' in res and 'cuda' in res and res['cpu']['pass'] and res['cuda']['pass'])
    P.write_json(path, res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate-ii')
    g.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'gate-ii':
        res = gate_ii(args.device)
        part = res[args.device]
        print(f'gate (ii) {args.device} part: pass={part["pass"]}; {part}; overall pass={res["pass"]}')
        return 0 if part['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
