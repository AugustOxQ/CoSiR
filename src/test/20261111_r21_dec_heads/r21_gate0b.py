"""Gate 0b (spec §3.5; ticket 03): r21's trainer with DEC off gives the frozen chain's losses to 1e-6. CPU. VERDICT CODE.

Every run is c*'s chain (`r21_run.run_cohort`: k11=40's graphs and data, c*'s loss and model patches, `smoke=True`: 2
epochs, 5,000 fit pairs per head, the full validation set). The reference runs the frozen `ct2_d2.run`; r21's runs have it
replaced by `r21_train.run`. All cohorts use the same c* arm name in separate run roots, so config comparisons need
no arm-name exclusion. All values compared are recorded from the live runs:
  * the train loss of every step and aspect: `ms_train.head_loss`'s return value in train mode (recorded by wrapping
    `rg_full.head_loss_for`, whose result the chain patches in), and the total loss each step backpropagates (a wrapper
    of `torch.Tensor.backward`, which the loop calls once per step on the sum over the heads, DEC terms included);
  * the full history entries except wall-clock `seconds`, the best epochs, the validation exports and their configs.

Part (i): fixed batches from the saved c* seed 0 weights (r10's `full/weights/k11=40`, loaded into the model's views in
emotion, style, genre order, so the shared trunk is genre's). Four runs record 12 steps: the frozen run, r21 `dec-off`,
and r21 plain DEC at lambda = 0 with warm-up 1, twice (step 1 is the first batch of every aspect, steps 1 to 10 are epoch
1, step 11 is the first batch with the DEC term in the loss: `HeadDec.block` is called once per head and adds 0.0).
Step 11 completes backward and the optimizer update, including the centres' AdamW group; step 12's losses observe
that update, then GateStop stops before step 12's backward. The k-means centre start is replaced by the first k rows
in the lambda = 0 runs only (the true DEC gradient is exactly 0 at lambda = 0; k-means on 43k x 2048 rows takes minutes
on a CPU).
  i-a   r21 dec-off equals the frozen run in steps 1 to 12, per aspect and total.
  i-b1  r21 DEC at lambda 0 equals the frozen run in steps 1 to 10 (warm-up, DEC not started), per aspect and total.
  i-b2  the DEC term adds exactly 0: each step's total equals the float32 left-to-right sum of its head losses,
        steps 1 to 12; and the first head's (emotion) step 11 loss equals the frozen run's. This checks the forward
        contribution through the first observed step after a DEC backward; dropout still shifts this run's losses.
  i-c   the same run with the torch RNG state restored after each `HeadDec.block` equals the frozen run in all 12 steps,
        per aspect and total. WHY: `HeadDec.block` is a separate train-mode forward pass, which draws dropout masks from the
        global torch RNG; at lambda = 0 the DEC term is 0 but the later heads' dropout in step 11, and every later step,
        see a shifted stream. So r21 with DEC on is not bit-equal to the frozen loop after DEC start, even at lambda = 0
        (`info` entry; i-c checks that RNG is the only difference, including after the first DEC backward and optimizer
        update, catching a forward-zero term with a leaked gradient). i-b1 and i-b2 are what holds as the trainer is.
Part (ii): the 2-epoch smoke from the chain's own initialisation: frozen and r21 `dec-off`.
  ii-a full history entries equal exactly except `seconds`; an r21-only key is allowed only if None;
  ii-b every step's losses and totals equal to TOL; ii-c best epochs and validation exports equal to TOL, with exactly
  one export per side and aspect and matching filenames; ii-d sidecar configs equal exactly except r21's added
  `dec` and `r21_code_sha256`.
Each check also needs its provenance: r21's runs called `r21_train.validate_dec` (so the replaced run was used), the
frozen runs did not; each lambda = 0 run reached the DEC block 6 times and returned exactly 0.0 every time. Each part
(i) run called the saved-weights builder once and loaded exactly one saved view per aspect (3 loads).

PASS/FAIL per check on stdout and a JSON record. Overall PASS requires both parts; successful single-part runs are
PARTIAL (exit 0). FAIL exits 1. With `--mutate`, FAIL prints "mutation detected" and exits 0; PASS or PARTIAL prints
"mutation NOT detected" and exits 1. `loss_scale` multiplies r21's batch loss by 1 + 1e-5; `lam_leak` runs the lambda = 0
runs at lambda = 1e-3; `grad_leak` adds exactly `1e-3*(z.sum() - z.sum().detach())` to their DEC block's return, using
the existing forward's z (no extra forward or RNG draws): each must give FAIL. The latter two need part (i).
Training outputs stay under /tmp/claude-0; --record-dir may also put only the JSON record under
/data/SSD2/cosir_r21/20261111_dec_heads/scores/ (by default the record goes into --outputs).

CLI: r21_gate0b.py --outputs DIR [--record-dir DIR] [--parts i ii] [--mutate loss_scale|lam_leak|grad_leak]
"""
import argparse
import json
import sys
import time
from contextlib import ExitStack
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
STEPS_I = 12                  # 10 warm-up batches, a DEC backward/update at 11, then observe its effect at 12
LOSS_SCALE = 1 + 1e-5
LAM_LEAK = 1e-3
FROZEN = 'k11=40'             # the frozen run's arm name (as rs_full.train)
SAVED = r21_run.R10_OUTPUTS / 'full' / 'weights' / 'k11=40'
SAVED_SEED = 0
LAM0 = dict(form='plain', lam=0.0, warmup=1, stop='S1', balance=False)
OFF = dict(form='off', lam=None, warmup=None, stop=None, balance=None)
RECORD_ROOT = r21_run.OUT_ROOT / 'scores'


class GateStop(Exception):
    """Raised inside the loop to end a run after `STEPS_I` steps."""


class Run:
    """What one run recorded: train losses per step and aspect, the totals, provenance counters, the outcome."""
    def __init__(self, label):
        self.label, self.steps, self.totals, self.validate_calls, self.block_calls, self.block_values = label, [], [], 0, 0, []
        self.outcome, self.root = None, None
        self.saved_builder_calls, self.saved_loads = 0, []

    def per_step(self):
        """[{aspect: loss}] in step order: each step records one loss per active head, in the aspects' order."""
        n = len(r21_run.ASPECTS)
        assert len(self.steps) % n == 0, (self.label, len(self.steps))
        return [dict(self.steps[i:i + n]) for i in range(0, len(self.steps), n)]


def saved_weights_loader(original, run):
    """`rb_student.build_model` plus c* seed 0's saved views (`torch.load(weights_only=True)`, as the inference code)."""
    def build(row, aspects, device, *, width):
        run.saved_builder_calls += 1
        models, views = original(row, aspects, device, width=width)
        for aspect in aspects:
            views[aspect].load_state_dict(torch.load(SAVED / f'{aspect}__s{SAVED_SEED}.pt', weights_only=True))
            run.saved_loads.append(aspect)
        return models, views
    return build


def one_run(ctx, label, root, settings, *, saved, stop_after=None, scale=1.0, stub_kmeans=False, rng_safe=False,
            grad_leak=False):
    """Train one cohort and record it. `settings` None: the frozen `ct2_d2.run`."""
    run = Run(label)
    run.root = Path(root)
    original_head_loss_for = rg_full.head_loss_for
    original_backward = torch.Tensor.backward
    original_block = r21_train.HeadDec.block
    original_validate = r21_train.validate_dec
    original_soft_assign = r21_dec.soft_assign

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
        if grad_leak:
            zs = []

            def soft_assign(z, *a, **kw):
                zs.append(z)
                return original_soft_assign(z, *a, **kw)

            # Capture the z from the existing forward inside HeadDec.block, without changing its RNG draws.
            with patch.object(r21_dec, 'soft_assign', soft_assign):
                term = original_block(hd, *args, **kwargs)
            assert len(zs) == 1, 'grad_leak needs the single z used by HeadDec.block'
            z = zs[0]
            term = term + 1e-3*(z.sum() - z.sum().detach())
        else:
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
            stack.enter_context(patch.object(rb_student, 'build_model', saved_weights_loader(rb_student.build_model, run)))
        if stub_kmeans:
            stack.enter_context(patch.object(r21_dec, 'kmeans_centres', lambda z, k, seed: z[:k].detach().clone().float()))
        dec_data = None if settings is None or settings['form'] == 'off' else {a: dict(k=r21_run.CLASS_COUNT[a]) for a in r21_run.ASPECTS}
        name = FROZEN  # the arm is recorded in config['rg']; separate roots distinguish the runs
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


def provenance(name, run, *, r21, blocks=0, saved=False):
    expected_loads = list(r21_run.ASPECTS) if saved else []
    ok = ((run.validate_calls == (1 if r21 else 0)) and run.block_calls == blocks
          and all(v == 0.0 for v in run.block_values)
          and run.saved_builder_calls == int(saved) and run.saved_loads == expected_loads)
    return dict(check=name, status='PASS' if ok else 'FAIL', max_abs_diff=None, n_values=None, tol=None,
                detail=f'{run.label}: validate_dec calls {run.validate_calls} (want {int(r21)}), DEC block calls '
                       f'{run.block_calls} (want {blocks}), block values {run.block_values}; saved-weights builder '
                       f'calls {run.saved_builder_calls} (want {int(saved)}), saved view loads {run.saved_loads} '
                       f'(want {expected_loads})')


def histories(run):
    """Full history entries, omitting only wall-clock seconds."""
    return {a: [{k: v for k, v in h.items() if k != 'seconds'} for h in run.outcome[a]['history']]
            for a in r21_run.ASPECTS}


def compare_histories(hf, ho):
    """Shared entries/keys must match exactly; r21-only keys must be None with DEC off."""
    errors = []
    for aspect in r21_run.ASPECTS:
        for i, (frozen, off) in enumerate(zip(hf[aspect], ho[aspect]), start=1):
            extra = off.keys() - frozen.keys()
            invalid = sorted(k for k in extra if off[k] is not None)
            common_off = {k: v for k, v in off.items() if k not in extra}
            if invalid or frozen != common_off:
                changed = sorted(k for k in frozen.keys() | common_off.keys()
                                 if k not in frozen or k not in common_off or frozen[k] != common_off[k])
                errors.append(f'{aspect} entry {i}: differing/missing keys {changed}, non-None r21-only keys {invalid}')
    return dict(check='ii-a full history entries except seconds, frozen vs r21 dec-off (3 heads x 2 epochs)',
                status='FAIL' if errors else 'PASS', max_abs_diff=None, n_values=sum(map(len, hf.values())), tol=None,
                detail='; '.join(errors))


def export_diff(root_a, root_b, arm_a, arm_b):
    """Export max abs diff, row count, and sidecar config comparison (excluding only r21's two added fields)."""
    diffs, rows, config_errors = [], 0, []
    for aspect in r21_run.ASPECTS:
        paths = []
        for root, arm in ((root_a, arm_a), (root_b, arm_b)):
            matches = sorted((Path(root) / 'train' / arm / rg_full.RECIPE).glob(f'{aspect}__*__s0.npz'))
            assert len(matches) == 1, (aspect, arm, 'expected exactly one export', matches)
            paths.append(matches[0])
        assert paths[0].name == paths[1].name, ('export filenames differ', *paths)
        row, configs = [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                row.append({side: z[side].astype(np.float64) for side in ('img', 'cap')})
            configs.append(json.loads(path.with_suffix('.json').read_text())['config'])
        frozen_config, off_config = configs
        off_config = {k: v for k, v in off_config.items() if k not in ('dec', 'r21_code_sha256')}
        if frozen_config != off_config:
            changed = sorted(k for k in frozen_config.keys() | off_config.keys()
                             if k not in frozen_config or k not in off_config or frozen_config[k] != off_config[k])
            config_errors.append(f'{aspect}: differing/missing config keys {changed}')
        for side in ('img', 'cap'):
            a, b = row[0][side], row[1][side]
            assert a.shape == b.shape == (6152, 2048), (aspect, side, a.shape, b.shape)
            diffs.append(float(np.max(np.abs(a - b))))
            rows += len(a)
    config_check = dict(check='ii-d validation sidecar configs, frozen vs r21 dec-off (except dec and r21_code_sha256)',
                        status='FAIL' if config_errors else 'PASS', max_abs_diff=None,
                        n_values=len(r21_run.ASPECTS), tol=None, detail='; '.join(config_errors))
    return max(diffs), rows, config_check


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
    out.append(compare('i-a r21 dec-off vs frozen: steps 1-12, per aspect and total',
                       flat_steps(frozen) + frozen.totals, flat_steps(off) + off.totals,
                       'step 1 = the first batch of each aspect'))
    n = 10 * len(r21_run.ASPECTS)
    out.append(compare('i-b1 r21 DEC at lambda 0 vs frozen: steps 1-10 (warm-up), per aspect and total',
                       flat_steps(frozen)[:n] + frozen.totals[:10], flat_steps(lam0)[:n] + lam0.totals[:10]))
    out.append(compare('i-b2 r21 DEC at lambda 0: total = sum of head losses in every step 1-12 (DEC adds exactly 0), '
                       'and step 11 emotion loss = frozen\'s',
                       [sum32(s) for s in lam0.per_step()] + [frozen.per_step()[10][first]],
                       lam0.totals + [lam0.per_step()[10][first]]))
    out.append(compare('i-c r21 DEC at lambda 0 with the RNG state restored after each DEC block vs frozen: steps 1-12, '
                       'per aspect and total', flat_steps(frozen) + frozen.totals, flat_steps(lam0_rng) + lam0_rng.totals,
                       'step 11 = the first DEC backward/update; step 12 observes its effect'))
    step11 = {a: lam0.per_step()[10][a] - frozen.per_step()[10][a] for a in r21_run.ASPECTS}
    out.append(dict(check='i-info lambda 0 as the trainer is, step 11 head loss minus frozen (not a gate)', status='INFO',
                    max_abs_diff=max(abs(v) for v in step11.values()), n_values=3, tol=None,
                    detail=f'{step11}: dropout masks of the later heads shifted by the DEC block\'s extra forward pass'))
    out.append(provenance('i-0 provenance: frozen', frozen, r21=False, saved=True))
    out.append(provenance('i-0 provenance: r21 dec-off', off, r21=True, saved=True))
    out.append(provenance('i-0 provenance: r21 lambda 0', lam0, r21=True, blocks=6, saved=True))
    out.append(provenance('i-0 provenance: r21 lambda 0, RNG restored', lam0_rng, r21=True, blocks=6, saved=True))
    return out


def check_ii(frozen, off):
    hf, ho = histories(frozen), histories(off)
    assert all(len(hf[a]) == len(ho[a]) == 2 for a in r21_run.ASPECTS), 'a 2-epoch smoke has 2 epochs per head'
    out = [compare_histories(hf, ho)]
    assert len(frozen.per_step()) == len(off.per_step()) == 20
    out.append(compare('ii-b every step\'s losses (20 steps) and totals, frozen vs r21 dec-off',
                       flat_steps(frozen) + frozen.totals, flat_steps(off) + off.totals))
    best = [frozen.outcome[a]['best_epoch'] for a in r21_run.ASPECTS]
    best_off = [off.outcome[a]['best_epoch'] for a in r21_run.ASPECTS]
    exp, rows, config_check = export_diff(frozen.root, off.root, FROZEN, FROZEN)
    check = compare('ii-c best epochs and validation exports (img, cap; 3 aspects), frozen vs r21 dec-off',
                    best + [exp], best_off + [0.0],
                    f'best epochs {best} vs {best_off}; export max abs diff {exp:.3g}, {rows} rows')
    out.append(check)
    out.append(config_check)
    out.append(provenance('ii-0 provenance: frozen', frozen, r21=False))
    out.append(provenance('ii-0 provenance: r21 dec-off', off, r21=True))
    return out


@threadpool_limits.wrap(limits=8)
def gate(outputs, parts, mutate, record_dir=None):
    outputs = Path(outputs).resolve()
    assert outputs.is_relative_to(r21_run.SMOKE_ROOT), f'gate 0b writes only under {r21_run.SMOKE_ROOT}: {outputs}'
    record_dir = outputs if record_dir is None else Path(record_dir).resolve()
    assert record_dir.is_relative_to(r21_run.SMOKE_ROOT) or record_dir.is_relative_to(RECORD_ROOT), \
        f'gate 0b records only under {r21_run.SMOKE_ROOT} or {RECORD_ROOT}: {record_dir}'
    parts = [part for part in ('i', 'ii') if part in parts]
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
                         stub_kmeans=True, grad_leak=mutate == 'grad_leak')
            l0r = one_run(ctx, 'lam0rng_i', work / 'lam0rng_i' / 'full', lam0, saved=True, stop_after=STEPS_I, scale=scale,
                          stub_kmeans=True, rng_safe=True, grad_leak=mutate == 'grad_leak')
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
    status = 'FAIL'
    if checks and all(c['status'] in ('PASS', 'INFO') for c in checks):
        status = 'PASS' if parts == ['i', 'ii'] else 'PARTIAL'
    commit, dirty = r21_run.git_state()
    record = dict(gate='0b', status=status, parts=parts, mutate=mutate, tol=TOL, checks=checks, work=str(work),
                  frozen_sha256=shas, code_sha256=rs_full.code_sha256(), code_commit=commit, code_dirty=dirty,
                  seconds=time.monotonic() - started, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f'r21_gate0b{"_mutate_" + mutate if mutate else ""}_{stamp}.json'
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'gate 0b: {status} (parts {", ".join(parts)}){" (mutation " + mutate + ")" if mutate else ""}; '
          f'record {path}; {record["seconds"]:.0f} s', flush=True)
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outputs', type=Path, required=True, help='a folder under /tmp/claude-0 (the smoke folder)')
    p.add_argument('--record-dir', type=Path, default=None,
                   help=f'JSON record only: under /tmp/claude-0 or {RECORD_ROOT}; default: --outputs')
    p.add_argument('--parts', nargs='+', choices=('i', 'ii'), default=['i', 'ii'])
    p.add_argument('--mutate', choices=('loss_scale', 'lam_leak', 'grad_leak'), default=None)
    args = p.parse_args(argv)
    status = gate(args.outputs, args.parts, args.mutate, record_dir=args.record_dir)
    if args.mutate:
        detected = status == 'FAIL'
        print('mutation detected' if detected else 'mutation NOT detected', flush=True)
        sys.exit(0 if detected else 1)
    sys.exit(1 if status == 'FAIL' else 0)


if __name__ == '__main__':
    main()
