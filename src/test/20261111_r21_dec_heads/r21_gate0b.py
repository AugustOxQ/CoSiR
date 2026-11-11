"""Gate 0b (spec §3.5; ticket 03): r21's trainer with DEC off gives the frozen chain's losses to 1e-6. CPU. VERDICT CODE.

Every run is c*'s chain (`r21_run.run_cohort`: k11=40's graphs and data, c*'s loss and model patches, `smoke=True`: 2
epochs, 5,000 fit pairs per head, the full validation set). The reference runs the frozen `ct2_d2.run`; r21's runs have it
replaced by `r21_train.run`. All values compared are recorded from the live runs:
  * the train loss of every step and aspect: `ms_train.head_loss`'s return value in train mode (recorded by wrapping
    `rg_full.head_loss_for`, whose result the chain patches in), and the total loss each step backpropagates (a wrapper
    of `torch.Tensor.backward`, which the loop calls once per step on the sum over the heads, DEC terms included);
  * the per-epoch train and dev losses of the histories, the best epochs, the validation exports.

Part (i): fixed batches from the saved c* seed 0 weights (r10's `full/weights/k11=40`, loaded into the model's views in
emotion, style, genre order, so the shared trunk is genre's). Four runs stop after step 11: the frozen run, r21 `dec-off`,
and r21 plain DEC at lambda = 0 with warm-up 1, twice (step 1 is the first batch of every aspect, steps 1 to 10 are epoch
1, step 11 is the first batch with the DEC term in the loss: `HeadDec.block` is called once per head and adds 0.0). The
k-means centre start is replaced by the first k rows in the lambda = 0 runs only (the centres cannot reach the loss at
lambda = 0; k-means on 43k x 2048 rows takes minutes on a CPU).
  i-a   r21 dec-off equals the frozen run in steps 1 to 11, per aspect and total.
  i-b1  r21 DEC at lambda 0 equals the frozen run in steps 1 to 10 (warm-up, DEC not started), per aspect and total.
  i-b2  at step 11 the DEC term adds exactly 0: each step's total equals the float32 left-to-right sum of its head losses,
        steps 1 to 11; and the first head's (emotion) step 11 loss equals the frozen run's.
  i-c   the same run with the torch RNG state restored after each `HeadDec.block` equals the frozen run in all 11 steps,
        per aspect and total. WHY: `HeadDec.block` is a separate train-mode forward pass, which draws dropout masks from the
        global torch RNG; at lambda = 0 the DEC term is 0 but the later heads' dropout in step 11, and every later step,
        see a shifted stream. So r21 with DEC on is not bit-equal to the frozen loop after DEC start, even at lambda = 0
        (`info` entry; i-c shows the RNG is the only difference). i-b1 and i-b2 are what holds as the trainer is.
Part (ii): the 2-epoch smoke from the chain's own initialisation: frozen and r21 `dec-off`.
  ii-a per-epoch train and dev losses of every head equal; ii-b every step's losses and totals equal;
  ii-c best epochs and validation exports equal.
Each check also needs its provenance: r21's runs called `r21_train.validate_dec` (so the replaced run was used), the
frozen runs did not; the lambda = 0 run reached the DEC block 3 times.

PASS/FAIL per check on stdout, a JSON record, exit 1 on any FAIL. `--mutate loss_scale` multiplies r21's batch loss by
1 + 1e-5, `--mutate lam_leak` runs the lambda = 0 run at lambda = 1e-3: each must give FAIL (the gate can fail).

CLI: r21_gate0b.py [--outputs DIR] [--parts i ii] [--mutate loss_scale|lam_leak]
"""
import argparse
import json
import sys
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import r21_run  # noqa: E402  (puts the chain's modules on sys.path)
import r21_dec  # noqa: E402
import r21_train  # noqa: E402
import rb_student  # noqa: E402
import rg_full  # noqa: E402
import rs_full  # noqa: E402
from ab_paths import ms_train  # noqa: E402

TOL = 1e-6                    # absolute, on losses (about 7 to 13) and on the float16 exports' values
STEPS_I = 11                  # part (i): epoch 1 (10 batches of 512 of the first 5,000 pairs) and the first DEC batch
LOSS_SCALE = 1 + 1e-5
LAM_LEAK = 1e-3
FROZEN = 'k11=40'             # the frozen run's arm name (as rs_full.train)
SAVED = r21_run.R10_OUTPUTS / 'full' / 'weights' / 'k11=40'
SAVED_SEED = 0
LAM0 = dict(form='plain', lam=0.0, warmup=1, stop='S1', balance=False)
OFF = dict(form='off', lam=None, warmup=None, stop=None, balance=None)


class GateStop(Exception):
    """Raised inside the loop to end a run after `STEPS_I` steps."""


class Run:
    """What one run recorded: train losses per step and aspect, the totals, provenance counters, the outcome."""
    def __init__(self, label):
        self.label, self.steps, self.totals, self.validate_calls, self.block_calls, self.block_values = label, [], [], 0, 0, []
        self.outcome, self.root = None, None

    def per_step(self):
        """[{aspect: loss}] in step order: each step records one loss per active head, in the aspects' order."""
        n = len(r21_run.ASPECTS)
        assert len(self.steps) % n == 0, (self.label, len(self.steps))
        return [dict(self.steps[i:i + n]) for i in range(0, len(self.steps), n)]


def saved_weights_loader(original):
    """`rb_student.build_model` plus c* seed 0's saved views (`torch.load(weights_only=True)`, as the inference code)."""
    def build(row, aspects, device, *, width):
        models, views = original(row, aspects, device, width=width)
        for aspect in aspects:
            views[aspect].load_state_dict(torch.load(SAVED / f'{aspect}__s{SAVED_SEED}.pt', weights_only=True))
        return models, views
    return build


def one_run(ctx, label, root, settings, *, saved, stop_after=None, scale=1.0, stub_kmeans=False, rng_safe=False):
    """Train one cohort and record it. `settings` None: the frozen `ct2_d2.run`."""
    run = Run(label)
    run.root = Path(root)
    original_head_loss_for = rg_full.head_loss_for
    original_backward = torch.Tensor.backward
    original_block = r21_train.HeadDec.block
    original_validate = r21_train.validate_dec

    def head_loss_for(*args, **kwargs):
        inner = original_head_loss_for(*args, **kwargs)

        def loss_fn(*a, **kw):
            loss = inner(*a, **kw)
            if scale != 1.0:
                loss = loss * scale
            if torch.is_grad_enabled():          # the train steps; the dev passes run under no_grad
                run.steps.append((kw['aspect'], float(loss.detach())))
            return loss
        return loss_fn

    def backward(self, *args, **kwargs):
        run.totals.append(float(self.detach()))
        if stop_after is not None and len(run.totals) >= stop_after:
            raise GateStop
        return original_backward(self, *args, **kwargs)

    def block(hd, *args, **kwargs):
        state = torch.get_rng_state() if rng_safe else None
        term = original_block(hd, *args, **kwargs)
        if rng_safe:
            torch.set_rng_state(state)
        run.block_calls += 1
        run.block_values.append(float(term.detach()))
        return term

    def validate(*args, **kwargs):
        run.validate_calls += 1
        return original_validate(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(rg_full, 'head_loss_for', head_loss_for))
        stack.enter_context(patch.object(torch.Tensor, 'backward', backward))
        stack.enter_context(patch.object(r21_train.HeadDec, 'block', block))
        stack.enter_context(patch.object(r21_train, 'validate_dec', validate))
        if saved:
            stack.enter_context(patch.object(rb_student, 'build_model', saved_weights_loader(rb_student.build_model)))
        if stub_kmeans:
            stack.enter_context(patch.object(r21_dec, 'kmeans_centres', lambda z, k, seed: z[:k].detach().clone().float()))
        dec_data = None if settings is None or settings['form'] == 'off' else {a: dict(k=r21_run.CLASS_COUNT[a]) for a in r21_run.ASPECTS}
        name = FROZEN if settings is None else f'gate0b-{label}'
        start = time.monotonic()
        try:
            run.outcome = r21_run.run_cohort(ctx, name, settings, 0, 'cpu', root, smoke=True, dec_data=dec_data)
        except GateStop:
            run.outcome = None
        print(f'  run {label}: {time.monotonic() - start:.0f} s, {len(run.totals)} steps', flush=True)
    return run


# ------------------------------------------------------------------------------------------------------- the checks
def compare(name, a, b, detail=''):
    """PASS when the two equal-length, non-empty, finite lists agree to TOL everywhere."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = a.shape == b.shape and a.size > 0 and bool(np.isfinite(a).all() and np.isfinite(b).all())
    diff = float(np.max(np.abs(a - b))) if ok else None
    ok = ok and diff <= TOL
    return dict(check=name, status='PASS' if ok else 'FAIL', max_abs_diff=diff, n_values=int(a.size), tol=TOL, detail=detail)


def flat_steps(run):
    return [step[a] for step in run.per_step() for a in r21_run.ASPECTS]


def provenance(name, run, *, r21, blocks=0):
    ok = (run.validate_calls == (1 if r21 else 0)) and run.block_calls == blocks and all(v == 0.0 for v in run.block_values)
    return dict(check=name, status='PASS' if ok else 'FAIL', max_abs_diff=None, n_values=None, tol=None,
                detail=f'{run.label}: validate_dec calls {run.validate_calls} (want {int(r21)}), DEC block calls '
                       f'{run.block_calls} (want {blocks}), block values {run.block_values}')


def histories(run):
    return {a: [(h['epoch'], h['train_loss'], h['dev_loss']) for h in run.outcome[a]['history']] for a in r21_run.ASPECTS}


def export_diff(root_a, root_b, arm_a, arm_b):
    """Max abs difference of the validation exports (img, cap; three aspects) of two runs, and the row counts compared."""
    diffs, rows = [], 0
    for aspect in r21_run.ASPECTS:
        row = {}
        for root, arm in ((root_a, arm_a), (root_b, arm_b)):
            path = next((Path(root) / 'train' / arm / rg_full.RECIPE).glob(f'{aspect}__*__s0.npz'))
            with np.load(path, allow_pickle=False) as z:
                row[arm] = {side: z[side].astype(np.float64) for side in ('img', 'cap')}
        for side in ('img', 'cap'):
            a, b = row[arm_a][side], row[arm_b][side]
            assert a.shape == b.shape == (6152, 2048), (aspect, side, a.shape, b.shape)
            diffs.append(float(np.max(np.abs(a - b))))
            rows += len(a)
    return max(diffs), rows


def sum32(step):
    """The loop's total for one step: the head losses added left to right in float32 (DEC terms of 0.0 add nothing)."""
    total = np.float32(0.0)
    for i, a in enumerate(r21_run.ASPECTS):
        total = np.float32(step[a]) if i == 0 else np.float32(total + np.float32(step[a]))
    return float(total)


def check_i(frozen, off, lam0, lam0_rng):
    out = []
    for run in (frozen, off, lam0, lam0_rng):
        assert len(run.per_step()) == len(run.totals) == STEPS_I, (run.label, len(run.per_step()), len(run.totals))
    first = r21_run.ASPECTS[0]
    out.append(compare('i-a r21 dec-off vs frozen: steps 1-11, per aspect and total',
                       flat_steps(frozen) + frozen.totals, flat_steps(off) + off.totals,
                       'step 1 = the first batch of each aspect'))
    n = 10 * len(r21_run.ASPECTS)
    out.append(compare('i-b1 r21 DEC at lambda 0 vs frozen: steps 1-10 (warm-up), per aspect and total',
                       flat_steps(frozen)[:n] + frozen.totals[:10], flat_steps(lam0)[:n] + lam0.totals[:10]))
    out.append(compare('i-b2 r21 DEC at lambda 0: total = sum of head losses in every step 1-11 (DEC adds exactly 0), '
                       'and step 11 emotion loss = frozen\'s',
                       [sum32(s) for s in lam0.per_step()] + [frozen.per_step()[10][first]],
                       lam0.totals + [lam0.per_step()[10][first]]))
    out.append(compare('i-c r21 DEC at lambda 0 with the RNG state restored after each DEC block vs frozen: steps 1-11, '
                       'per aspect and total', flat_steps(frozen) + frozen.totals, flat_steps(lam0_rng) + lam0_rng.totals,
                       'step 11 = the first batch with the DEC term'))
    step11 = {a: lam0.per_step()[10][a] - frozen.per_step()[10][a] for a in r21_run.ASPECTS}
    out.append(dict(check='i-info lambda 0 as the trainer is, step 11 head loss minus frozen (not a gate)', status='INFO',
                    max_abs_diff=max(abs(v) for v in step11.values()), n_values=3, tol=None,
                    detail=f'{step11}: dropout masks of the later heads shifted by the DEC block\'s extra forward pass'))
    out.append(provenance('i-0 provenance: frozen', frozen, r21=False))
    out.append(provenance('i-0 provenance: r21 dec-off', off, r21=True))
    out.append(provenance('i-0 provenance: r21 lambda 0', lam0, r21=True, blocks=3))
    out.append(provenance('i-0 provenance: r21 lambda 0, RNG restored', lam0_rng, r21=True, blocks=3))
    return out


def check_ii(frozen, off):
    hf, ho = histories(frozen), histories(off)
    assert all(len(hf[a]) == len(ho[a]) == 2 for a in r21_run.ASPECTS), 'a 2-epoch smoke has 2 epochs per head'
    assert all([e for e, _, _ in hf[a]] == [e for e, _, _ in ho[a]] for a in r21_run.ASPECTS)
    flat = lambda h: [v for a in r21_run.ASPECTS for _, t, d in h[a] for v in (t, d)]  # noqa: E731
    out = [compare('ii-a per-epoch train and dev losses, frozen vs r21 dec-off (3 heads x 2 epochs)', flat(hf), flat(ho))]
    assert len(frozen.per_step()) == len(off.per_step()) == 20
    out.append(compare('ii-b every step\'s losses (20 steps) and totals, frozen vs r21 dec-off',
                       flat_steps(frozen) + frozen.totals, flat_steps(off) + off.totals))
    best = [frozen.outcome[a]['best_epoch'] for a in r21_run.ASPECTS]
    best_off = [off.outcome[a]['best_epoch'] for a in r21_run.ASPECTS]
    exp, rows = export_diff(frozen.root, off.root, FROZEN, f'gate0b-{off.label}')
    check = compare('ii-c best epochs and validation exports (img, cap; 3 aspects), frozen vs r21 dec-off',
                    best + [exp], best_off + [0.0], f'best epochs {best} vs {best_off}; export max abs diff {exp:.3g}')
    out.append(check)
    out.append(provenance('ii-0 provenance: frozen', frozen, r21=False))
    out.append(provenance('ii-0 provenance: r21 dec-off', off, r21=True))
    return out


@threadpool_limits.wrap(limits=8)
def gate(outputs, parts, mutate):
    outputs = Path(outputs).resolve()
    assert outputs.is_relative_to(r21_run.SMOKE_ROOT), f'gate 0b writes only under {r21_run.SMOKE_ROOT}: {outputs}'
    stamp = time.strftime('%Y%m%d_%H%M%S')
    work = outputs / f'gate0b_{stamp}'
    torch.set_num_threads(8)
    r21_run.rs_paths.check_imports()
    shas = rs_full.require_frozen_code()
    before = r21_run.tree_listing()
    started, checks = time.monotonic(), []
    scale = LOSS_SCALE if mutate == 'loss_scale' else 1.0
    lam0 = dict(LAM0, lam=LAM_LEAK if mutate == 'lam_leak' else 0.0)
    with ExitStack() as stack:
        ctx = r21_run.load_chain(stack)
        print(f'data loaded: {time.monotonic() - started:.0f} s', flush=True)
        if 'i' in parts:
            frozen = one_run(ctx, 'frozen_i', work / 'frozen_i' / 'full', None, saved=True, stop_after=STEPS_I)
            off = one_run(ctx, 'off_i', work / 'off_i' / 'full', OFF, saved=True, stop_after=STEPS_I, scale=scale)
            l0 = one_run(ctx, 'lam0_i', work / 'lam0_i' / 'full', lam0, saved=True, stop_after=STEPS_I, scale=scale,
                         stub_kmeans=True)
            l0r = one_run(ctx, 'lam0rng_i', work / 'lam0rng_i' / 'full', lam0, saved=True, stop_after=STEPS_I, scale=scale,
                          stub_kmeans=True, rng_safe=True)
            checks += check_i(frozen, off, l0, l0r)
        if 'ii' in parts:
            frozen = one_run(ctx, 'frozen_ii', work / 'frozen_ii' / 'full', None, saved=False)
            off = one_run(ctx, 'off_ii', work / 'off_ii' / 'full', OFF, saved=False, scale=scale)
            checks += check_ii(frozen, off)
    assert r21_run.tree_listing() == before, 'r10\'s tree changed during the gate (it is only read)'
    for c in checks:
        print(f'{c["status"]}  {c["check"]}'
              + (f'  (max abs diff {c["max_abs_diff"]:.3g}' + (f', tol {c["tol"]:g})' if c['tol'] is not None else ')')
                 if c['max_abs_diff'] is not None else '')
              + (f'  [{c["detail"]}]' if c['status'] != 'PASS' and c['detail'] else ''), flush=True)
    status = 'PASS' if checks and all(c['status'] in ('PASS', 'INFO') for c in checks) else 'FAIL'
    commit, dirty = r21_run.git_state()
    record = dict(gate='0b', status=status, parts=parts, mutate=mutate, tol=TOL, checks=checks, work=str(work),
                  frozen_sha256=shas, code_sha256=rs_full.code_sha256(), code_commit=commit, code_dirty=dirty,
                  seconds=time.monotonic() - started, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    path = outputs / f'r21_gate0b{"_mutate_" + mutate if mutate else ""}_{stamp}.json'
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'gate 0b: {status}{" (mutation " + mutate + ")" if mutate else ""}; record {path}; {record["seconds"]:.0f} s', flush=True)
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outputs', type=Path, required=True, help='a folder under /tmp/claude-0 (the smoke folder)')
    p.add_argument('--parts', nargs='+', choices=('i', 'ii'), default=['i', 'ii'])
    p.add_argument('--mutate', choices=('loss_scale', 'lam_leak'), default=None)
    args = p.parse_args(argv)
    sys.exit(0 if gate(args.outputs, args.parts, args.mutate) == 'PASS' else 1)


if __name__ == '__main__':
    main()
