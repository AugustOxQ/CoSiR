"""V2 for stage i (ticket 06; rule §4.1 with the RB rule §4.2): the part 2 statistic of K5 to K8 re-derived by
`rb_v2`'s separate code from the rule's text, the held list, the captions, the CT's records and the saved outputs.

`rb_v2.run_stage` is reused as it is (labels, R-COL eligibility, partitions, AMI, bootstrap, comparison); this module
adds what is this round's: the candidate list (K6, K7, K8, K5 with a block that has an outcome), the place of B1 and
B6 on ArtELingo-28 at the full scale (12-24's saved outputs; the sample scale is this round's) and this round's
header. Every run derives every candidate again (nothing is kept from an earlier file: an entry could outlive changed
outputs or code). It reads our verdict file only to compare, after its own numbers exist. Writes
`<outputs>/part2/rederive_v2_stage{3,4,5}.json`.

CLI (from the round folder):
  python eg_v2.py --stage 3|4|5 [--candidates K6 K7 ...] [--outputs DIR] [--smoke] [--workers 8]
  (--candidates only with --outputs under /tmp/claude-0/: a list would drop the other candidates from the file)
"""
import contextlib
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import eg_paths
import eg_encoders

eg_encoders.extend_rb_maps()
import rb_v2  # noqa: E402

OLD_CANDIDATES = ('K1', 'K2', 'K3', 'K4')       # 12-24's: only for the parity check on its saved outputs


@dataclasses.dataclass
class Roots(rb_v2.Roots):
    """`rb_v2.Roots` plus `frozen`: where 12-24's full-scale B1 and B6 outputs on ArtELingo-28 are read (this round
    computes B1 and B6 on ArtELingo-28 at the sample scale only; ticket 04)."""
    frozen: Path = eg_paths.RB_OUT

    @classmethod
    def default(cls, outputs=None, frozen=None):
        roots = super().default(outputs or eg_paths.OUTPUTS)
        roots.frozen = Path(frozen or eg_paths.RB_OUT)
        return roots


_RB_ARM_PLANS = rb_v2.arm_plans


def arm_plans(cand, mode, scale, roots):
    """`rb_v2.arm_plans`, with B1 and B6 of the full-scale ArtELingo-28 run read from `roots.frozen`."""
    plans = _RB_ARM_PLANS(cand, mode, scale, roots)
    if mode == 'a28' and scale == 'full':
        for plan in plans:
            if plan['name'] in ('B1', 'B6'):
                plan['base'] = Path(getattr(roots, 'frozen', eg_paths.RB_OUT)) / 'part2' / scale / mode
    return plans


@contextlib.contextmanager
def scoped_plans():
    """`rb_v2.arm_plans` replaced by `arm_plans` for the length of the block (rb_v2 is not edited)."""
    rb_v2.arm_plans = arm_plans
    try:
        yield
    finally:
        rb_v2.arm_plans = _RB_ARM_PLANS


def candidates_with_outcome(verdict_path, requested=None, allow_old=False):
    """(ready, skipped): `ready` are the candidates whose block in the verdict file has an outcome, in the rule's order
    (K6, K7, K8, K5); `skipped` maps a block without an outcome to {status, v_failure}. By default every block of
    K5 to K8 in the file; `requested` names them, and each needs a block. 12-24's candidates are refused unless
    `allow_old` (the parity check on its saved outputs)."""
    allowed = eg_encoders.CANDIDATES + (OLD_CANDIDATES if allow_old else ())
    present = [c for c in rb_v2._blocks_in(verdict_path) if c in allowed]
    if requested:
        assert all(c in allowed for c in requested), f'not a candidate of stage i: {sorted(set(requested) - set(allowed))}'
        missing = [c for c in requested if c not in present]
        assert not missing, f'{verdict_path}: no block for {missing}'
        present = list(requested)
    order = {c: i for i, c in enumerate(allowed)}
    ready, skipped = [], {}
    for cand in sorted(present, key=order.__getitem__):
        outcome_id, v_failure = rb_v2.outcome_of(verdict_path, cand)
        if outcome_id is None:
            skipped[cand] = dict(status='no outcome, not re-derived', v_failure=v_failure)
        else:
            ready.append(cand)
    return ready, skipped


def _write_json(path, doc):
    tmp = Path(str(path) + '.pending')
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)


def run_stage(stage, outputs=None, candidates=None, draws=rb_v2.DRAWS, workers=8, smoke=False, roots=None,
              allow_old=False):
    """Re-derive each candidate of `part2/verdict_stage{stage}.json` that has an outcome and compare; the file is
    written whole, every candidate derived again. `candidates` (a subset) only for outputs under /tmp/claude-0/:
    the file would lack the others. Returns {agree, file, candidates: {c: mismatches}, skipped: {c: v_failure}}."""
    outputs = Path(outputs or eg_paths.OUTPUTS)
    roots = roots or Roots.default(outputs)
    assert Path(roots.outputs) == outputs, (roots.outputs, outputs)
    verdict = outputs / f'part2/verdict_stage{stage}.json'
    assert verdict.exists(), f'{verdict} does not exist: V2 runs after the verdict file'
    assert not candidates or eg_paths.smoke_ok(outputs), \
        '--candidates (a subset) only with outputs under /tmp/claude-0/: the file would lack the other candidates'
    if smoke:
        assert eg_paths.smoke_ok(outputs), 'smoke needs --outputs under /tmp/claude-0/'
        draws = 20
    ready, skipped = candidates_with_outcome(verdict, candidates, allow_old)
    target = outputs / f'part2/rederive_v2_stage{stage}.json'
    if ready:
        with scoped_plans():
            rb_v2.run_stage(stage, roots, candidates=ready, draws=draws, workers=workers, smoke=smoke)
        doc = json.loads(target.read_text())
        written = doc['candidates']
        assert list(written) == ready, (list(written), ready)
    else:
        written = {}
        doc = dict(schema=1, stage=stage, smoke=smoke, tolerance=rb_v2.TOL, verdict_file=str(verdict))
    agree = bool(written) and all(r['comparison']['agree'] for r in written.values())   # nothing checked is no agreement
    if not written:
        print(f'V2 stage {stage}: no candidate has an outcome; nothing re-derived (stage stops)', flush=True)
    doc.update(agree=agree, rule=str(eg_paths.RULE), rb_rule=str(rb_v2.rb_paths.RULE), round='encoders_genre',
               code_commit=_git_head(), eg_v2_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               candidates=written, skipped=skipped)
    _write_json(target, doc)
    return dict(agree=agree, file=str(target), candidates={c: r['comparison']['mismatches'] for c, r in written.items()},
                skipped={c: v['v_failure'] for c, v in skipped.items()})


def _git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip()


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', type=int, choices=(3, 4, 5), required=True)
    p.add_argument('--candidates', nargs='+', choices=eg_encoders.CANDIDATES + OLD_CANDIDATES,
                   help='default (and the only form for the real outputs): every block of K5 to K8 with an outcome. '
                        'A list, and K1 to K4 (the parity check on 12-24\'s saved outputs), only with --outputs '
                        'under /tmp/claude-0/')
    p.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true', help='20 draws; outputs under /tmp/claude-0/')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args(argv)
    assert not args.candidates or eg_paths.smoke_ok(args.outputs), \
        '--candidates (a subset) only with --outputs under /tmp/claude-0/: the file would lack the other candidates'
    old = bool(args.candidates) and any(c in OLD_CANDIDATES for c in args.candidates)
    assert not old or eg_paths.smoke_ok(args.outputs), 'K1 to K4 only with --outputs under /tmp/claude-0/'
    summary = run_stage(args.stage, args.outputs, args.candidates, workers=args.workers, smoke=args.smoke,
                        allow_old=old)
    print(json.dumps(summary, indent=1))
    return 0 if (summary['agree'] or args.smoke) else 1


if __name__ == '__main__':
    sys.exit(main())
