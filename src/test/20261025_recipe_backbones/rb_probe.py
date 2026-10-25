"""Ceiling probe (spec §3.6, D14, D34; literature note 12-20, idea 1): can the default image predict the instructed one?

Per aspect (emotion -> emotion_focus, style -> style_word, genre -> genre_word) a ridge model and a 4-layer MLP predict
the aspect's instructed image from the default image. Both fit on the 13,400 sample paintings of the fit role; the
ridge alpha and the MLP's stopping epoch are chosen on the 1,489 stop paintings' regression loss (no labels). Inputs
and targets are standardized per dimension with the fit rows' mean and standard deviation. Loss (both models, both
sets): mean of (1 - cosine) and smooth-L1, between the prediction and the standardized target. Seed 0.
Predictions are mapped back to the target scale, L2-normalized, then clustered as in part 1 (`rb_ami.partitions`, all
rows clustered before masking) and scored by `rb_ami.score` on each set's labels. Sets: `P` (29,232) and `old5000`.
Beside them: the instructed and the default images, and R0-5's P and old-5,000 image exports of part 1
(`part1/train/R0-5/<aspect>__s0__<set>.npz`); a missing export is reported as missing, not an error.
Writes `part1/probe.json` and `part1/probe.md`. Exploration only: decides nothing.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import rb_paths
import rb_data
import rb_ami

ASPECTS = rb_paths.RB_ASPECTS
KEYS = {'emotion': 'emotion_focus', 'style': 'style_word', 'genre': 'genre_word'}
SETS = ('P', 'old5000')
SEED = 0
HIDDEN = 1024
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 0.05
BATCH = 512
MAX_EPOCHS = 200
PATIENCE = 15
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)  # alpha / n_fit, on standardized inputs
SMOKE_ROWS = 1000
SMOKE_EPOCHS = 2
THREADS = 8
REFERENCE = 'R0-5'
_CODE = ('rb_probe.py', 'rb_data.py', 'rb_ami.py')


def pick_device():
    """cuda only when CUDA_VISIBLE_DEVICES names a device (the run queue sets it under the GPU lock)."""
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class Standardizer:
    def __init__(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-6

    def fwd(self, x):
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inv(self, z):
        return (np.asarray(z, dtype=np.float64) * self.std + self.mean).astype(np.float32)


def loss_fn(pred, target):
    """Mean of (1 - cosine) and smooth-L1 (beta 1) between standardized prediction and target; torch tensors."""
    return 0.5 * ((1 - F.cosine_similarity(pred, target, dim=1)).mean() + F.smooth_l1_loss(pred, target))


def ridge_fit(xs, ys, grid=RIDGE_GRID):
    """Ridge on standardized arrays, one eigendecomposition for the whole grid -> {g: weights [d_in + 1, d_out]}."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = len(x)
    ym = y.mean(0)
    evals, evecs = np.linalg.eigh(x.T @ x)
    xty = evecs.T @ (x.T @ (y - ym))
    out = {}
    for g in grid:
        w = evecs @ (xty / (evals + g * n)[:, None])
        out[g] = np.vstack([w, ym[None, :]]).astype(np.float32)
    return out


def ridge_predict(w, xs):
    return np.asarray(xs, dtype=np.float32) @ w[:-1] + w[-1]


def make_mlp(d_in, d_out):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(HIDDEN, d_out))


def _predict(model, xs, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(xs[i:i + 4096]).to(device)).cpu() for i in range(0, len(xs), 4096)]).numpy()


def mlp_fit(xf, yf, xv, yv, device, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Train on (xf, yf), stop on the loss of (xv, yv). Returns (model with the best state, record)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_mlp(xf.shape[1], yf.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xt, yt = torch.from_numpy(xf).to(device), torch.from_numpy(yf).to(device)
    xvt, yvt = torch.from_numpy(xv).to(device), torch.from_numpy(yv).to(device)
    best, best_state, best_epoch, hist = np.inf, None, -1, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(xf))).to(device)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xvt), yvt))
        hist.append(val)
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_epoch + 1, epochs_run=len(hist), stop_loss=best, history=[round(h, 5) for h in hist])


def np_loss(pred, target):
    return float(loss_fn(torch.from_numpy(np.asarray(pred, dtype=np.float32)), torch.from_numpy(target)))


def probe_aspect(fit, stop, device, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """fit, stop: (default image, instructed image), unit float32. Returns (predict(default) -> unit float32 per model, record)."""
    for x, y in (fit, stop):
        assert x.shape[0] == y.shape[0] and x.ndim == 2 and y.ndim == 2
    sx, sy = Standardizer(fit[0]), Standardizer(fit[1])
    xf, yf, xv, yv = sx.fwd(fit[0]), sy.fwd(fit[1]), sx.fwd(stop[0]), sy.fwd(stop[1])
    t0 = time.time()
    ws = ridge_fit(xf, yf)
    losses = {g: np_loss(ridge_predict(w, xv), yv) for g, w in ws.items()}
    g_best = min(losses, key=losses.get)
    rec = {'ridge': dict(alpha_over_n=g_best, alpha=g_best * len(xf), grid_stop_loss={str(g): v for g, v in losses.items()},
                         stop_loss=losses[g_best], seconds=round(time.time() - t0, 1))}
    t0 = time.time()
    model, mrec = mlp_fit(xf, yf, xv, yv, device, max_epochs, patience)
    mrec.update(seconds=round(time.time() - t0, 1), device=str(device))
    rec['mlp'] = mrec

    def predict(x):
        z = sx.fwd(x)
        return {'ridge': unit(sy.inv(ridge_predict(ws[g_best], z))), 'mlp': unit(sy.inv(_predict(model, z, device)))}
    return predict, rec


def ami_row(x, labels, idx):
    """AMI (mean of 3 k-means partitions) of `x` (all rows clustered) on the labelled rows `idx`."""
    return rb_ami.score(labels, rb_ami.partitions(x), idx)


def read_reference(outputs, aspect, sname, shape):
    """R0-5's image export of part 1 for one set, or None when it is missing."""
    f = Path(outputs) / 'part1' / 'train' / REFERENCE / f'{aspect}__s0__{sname}.npz'
    if not f.exists():
        return None
    if sname == 'P':          # rows in P.json order: the export's training record names the same P (C6)
        p_sha = json.loads((Path(outputs) / 'part1' / 'P.json').read_text())['sha256']
        record = json.loads((f.parent / f'{aspect}__s0.json').read_text())
        assert record['p_sha256'] == p_sha, f'{f}: its record names another P'
    with np.load(f) as z:
        img = z['img']
    assert img.shape == tuple(shape), (f, img.shape)
    return img


def score_sets(aspect, sets, predict, reference):
    """sets {name: dict(default, instructed, labels)}; reference {name: array or None} -> {set: {view: ami}}."""
    out = {}
    for sname, s in sets.items():
        y = s['labels'][aspect]
        idx = np.flatnonzero(y >= 0)
        views = {'instructed': s['instructed'], 'default': s['default'], **predict(s['default'])}
        res = {k: ami_row(v, y, idx) for k, v in views.items()}
        ref = reference.get(sname)
        res[REFERENCE] = None if ref is None else ami_row(ref, y, idx)
        gap = res['instructed'] - res['default']
        res['ridge_fraction_of_gap'] = (res['ridge'] - res['default']) / gap if abs(gap) > 1e-9 else None
        res['mlp_fraction_of_gap'] = (res['mlp'] - res['default']) / gap if abs(gap) > 1e-9 else None
        res['labelled'] = int(len(idx))
        out[sname] = res
    return out


# ------------------------------------------------------------------ real data
def load_real(outputs):
    """-> fit/stop {aspect: (default, instructed)}, sets {name: dict(default, instr {aspect: arr}, labels)}, reference."""
    sdef = rb_data.sample_default_images()
    sins = {a: rb_data.sample_instructed_images(KEYS[a]) for a in ASPECTS}
    fit = {a: (rb_data.select_role('fit', sdef), rb_data.select_role('fit', sins[a])) for a in ASPECTS}
    stop = {a: (rb_data.select_role('stop', sdef), rb_data.select_role('stop', sins[a])) for a in ASPECTS}
    sets = {'P': dict(default=rb_data.p_default_images(), instr={a: rb_data.p_instructed_images(KEYS[a]) for a in ASPECTS},
                      labels=rb_data.p_labels()),
            'old5000': dict(default=rb_data.select_role('old5000', sdef),
                            instr={a: rb_data.select_role('old5000', sins[a]) for a in ASPECTS},
                            labels=rb_data.old5000_labels())}
    return fit, stop, sets


def subsample_rows(n, rows):
    return np.arange(0, n, max(1, n // rows))[:rows]


def run(fit, stop, sets, outputs, smoke=False):
    """The probe over all aspects -> the probe dictionary."""
    device = pick_device()
    torch.set_num_threads(THREADS)
    max_epochs = SMOKE_EPOCHS if smoke else MAX_EPOCHS
    sel = {}
    if smoke:
        fit = {a: (x[:SMOKE_ROWS], y[:SMOKE_ROWS]) for a, (x, y) in fit.items()}
        stop = {a: (x[:SMOKE_ROWS // 4], y[:SMOKE_ROWS // 4]) for a, (x, y) in stop.items()}
        for sname, s in sets.items():
            sel[sname] = subsample_rows(len(s['default']), SMOKE_ROWS)
    out = {}
    for aspect in ASPECTS:
        t0 = time.time()
        predict, rec = probe_aspect(fit[aspect], stop[aspect], device, max_epochs)
        view, reference = {}, {}
        for sname, s in sets.items():
            ref = read_reference(outputs, aspect, sname, s['default'].shape)
            i = sel.get(sname)
            view[sname] = dict(default=s['default'] if i is None else s['default'][i],
                               instructed=s['instr'][aspect] if i is None else s['instr'][aspect][i],
                               labels={a: (y if i is None else y[i]) for a, y in s['labels'].items()})
            reference[sname] = ref if (ref is None or i is None) else ref[i]
        rec['ami'] = score_sets(aspect, view, predict, reference)
        rec['n_fit'], rec['n_stop'] = len(fit[aspect][0]), len(stop[aspect][0])
        rec['seconds'] = round(time.time() - t0, 1)
        out[aspect] = rec
        print(f'{aspect}: {rec["seconds"]} s, mlp best epoch {rec["mlp"]["best_epoch"]}', flush=True)
    return {'smoke': bool(smoke), 'seed': SEED, 'device': str(device), 'sets': {k: int(len(sel[k]) if k in sel else len(v['default'])) for k, v in sets.items()},
            'config': dict(hidden=HIDDEN, layers=4, dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY, batch=BATCH,
                           max_epochs=max_epochs, patience=PATIENCE, ridge_grid_alpha_over_n=list(RIDGE_GRID),
                           loss='0.5*(1-cosine) + 0.5*smooth_l1, standardized targets', reference=REFERENCE),
            'versions': rb_ami.versions(), 'code_sha256': code_sha256(), 'aspects': out}


# ------------------------------------------------------------------ output
def code_sha256():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in _CODE}


def fmt(x):
    return 'missing' if x is None else f'{x:.4f}'


def render_md(probe):
    lines = ['# Ceiling probe', '', f"smoke: {probe['smoke']}; device {probe['device']}; sets {probe['sets']}. "
             'AMI of k-means (k 20, 3 states) on the labels. Exploration only; decides nothing.', '']
    for sname in SETS:
        lines += [f'## {sname}', '', f'| aspect | labelled | default | ridge | MLP | instructed | {REFERENCE} | ridge frac. | MLP frac. |',
                  '|---|---|---|---|---|---|---|---|---|']
        for a, r in probe['aspects'].items():
            m = r['ami'][sname]
            lines.append(f"| {a} | {m['labelled']} | {fmt(m['default'])} | {fmt(m['ridge'])} | {fmt(m['mlp'])} | "
                         f"{fmt(m['instructed'])} | {fmt(m[REFERENCE])} | {fmt(m['ridge_fraction_of_gap'])} | {fmt(m['mlp_fraction_of_gap'])} |")
        lines.append('')
    lines += ['frac. = (probe - default) / (instructed - default): near 1, the default image holds the information '
              '(losses are the limit); near 0, it is missing.', '', '| aspect | ridge alpha/n | ridge stop loss | MLP best epoch | epochs run | MLP stop loss |',
              '|---|---|---|---|---|---|']
    for a, r in probe['aspects'].items():
        lines.append(f"| {a} | {r['ridge']['alpha_over_n']} | {r['ridge']['stop_loss']:.4f} | {r['mlp']['best_epoch']} | "
                     f"{r['mlp']['epochs_run']} | {r['mlp']['stop_loss']:.4f} |")
    return '\n'.join(lines) + '\n'


def write_probe(outputs, probe):
    folder = Path(outputs) / 'part1'
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (('probe.json', json.dumps(probe, indent=1) + '\n'), ('probe.md', render_md(probe))):
        tmp = folder / f'{name}.pending'
        tmp.write_text(text)
        tmp.replace(folder / name)
    return folder / 'probe.json'


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write part1/probe.json and probe.md (ceiling probe).')
    ap.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    ap.add_argument('--smoke', action='store_true', help='1,000 rows, 2 epochs; outputs under /tmp/claude-0/')
    args = ap.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    rb_paths.check_imports()
    p = Path(args.outputs) / 'part1' / 'P.json'
    if p.exists():
        rb_data.read_p_json(args.outputs, expect=rb_data.p_names())
    fit, stop, sets = load_real(args.outputs)
    print(write_probe(args.outputs, run(fit, stop, sets, args.outputs, smoke=args.smoke)))


if __name__ == '__main__':
    sys.exit(main())
