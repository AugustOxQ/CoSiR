"""Stage ii P inference (rule §2.2, §4.2): every stage ii arm on the 29,232 P paintings from saved weights, and
VL1 (b), VL1 (c), VL3, VL4.

  infer  writes `stage_ii/p/<arm>/...` (`eg_paths.p_student`, `eg_paths.p_own`, each with a `.json` sidecar) and the index
         `stage_ii/p/index.json` (key `missing`: arm -> reason, for candidates without a stage 3 outcome or weights).
  check  VL1 (b), VL1 (c), VL3 on the exports; writes `stage_ii/vl1.json` (`pass`, items `b`, `c`, `vl3`).
  all    infer, then check.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import eg_encoders
import eg_run
import eg_paths  # noqa: F401  (sys.path)

STAGE = 'stage_ii'
SCALE = 'sample'
N_P = 29232
NORM_TOL = 1e-3
BATCH = 512                    # rb_student.infer's batch (ms_train.CONFIG['batch_size'])
OLD_STUDENTS = ('K4', 'K2', 'K3')   # 12-24's weights (RB_PART2); the others are this round's
SIDECAR_KEYS_STUDENT = ('p_sha256', 'weights_sha256', 'store', 'batch')


# ------------------------------------------------------------------ small helpers
def array_sha256(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def code_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=eg_paths.ROUND, capture_output=True, text=True,
                          check=True).stdout.strip()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **arrays)
    pending.replace(path)


def weights_root(outputs, student):
    """12-24's frozen sample root for K4, K2, K3; this round's for K5 to K8."""
    return (eg_paths.RB_PART2 if student in OLD_STUDENTS else Path(outputs) / 'part2') / SCALE


def own_checkpoint(outputs, own, seed):
    """B1: the CT's sample-scale core P3 mapper; M2, M3: 12-24's; M5 to M8: this round's."""
    if own == 'B1':
        return eg_paths.CT_OUT / f'percept/core/stage2/k40/P3__s{seed}.npz'
    root = eg_paths.RB_PART2 if own in ('M2', 'M3') else Path(outputs) / 'part2'
    return root / SCALE / f'matched/{own}/P3__s{seed}.npz'


def student_files(outputs, enc, aspects, seeds):
    import rb_student
    return [rb_student.weights_path(weights_root(outputs, enc.student), enc.student, a, s)
            for s in seeds for a in aspects]


def own_files(outputs, enc, seeds):
    files = []
    for s in seeds:
        ck = own_checkpoint(outputs, enc.own, s)
        files += [ck, ck.with_suffix('.pt'), ck.with_suffix('.json')]
    return files


def plan(outputs, aspects=eg_paths.ASPECTS, seeds=eg_paths.SEEDS, encoders=eg_encoders.ENCODERS):
    """(arms to infer, missing): an old arm without weights is an error; a new arm needs a stage 3 outcome and weights."""
    arms, missing = [], {}
    deferred = eg_run.deferred(outputs)
    stopped = eg_run.stage_i_failure(outputs)
    for enc in encoders:
        reason = ((stopped or ('deferred before stage ii' if enc.student in deferred else None)
                   or eg_run.stage3_failure(outputs, enc.student)) if enc.new else None)
        if reason:
            missing.update({enc.student: reason, enc.own: reason})
            continue
        for arm, files in ((enc.student, student_files(outputs, enc, aspects, seeds)),
                           (enc.own, own_files(outputs, enc, seeds))):
            absent = [f for f in files if not Path(f).is_file()]
            if absent and enc.new:
                relative = Path(absent[0]).relative_to(Path(outputs))
                missing[arm] = f'no stage 3 weights: {relative} (and {len(absent) - 1} more) not found'
            else:
                assert not absent, f'{arm}: saved weights missing: {absent[:3]}'
                arms.append(arm)
    return arms, missing


# ------------------------------------------------------------------ VL3 on one export
def vl3_student(img, n_rows):
    assert img.dtype == np.float16 and img.shape == (n_rows, 2048), (img.dtype, img.shape)
    assert np.isfinite(img).all(), 'nonfinite rows'
    err = float(np.abs(np.linalg.norm(img.astype(np.float32), axis=1) - 1).max())
    assert err < NORM_TOL, f'norm error {err}'
    return err


def vl3_own(prob, n_rows):
    assert prob.dtype == np.float32 and prob.shape == (n_rows, 40), (prob.dtype, prob.shape)
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, 'not sigmoid probabilities'
    return 0.0


def check_export(path, p_sha, n_rows, kind):
    """VL3 for a file: sidecar P hash, finite rows, shape, unit norm (students). Returns (ok, detail dict)."""
    path = Path(path)
    detail = dict(file=str(path))
    try:
        side = json.loads(path.with_suffix('.json').read_text())
        assert side['p_sha256'] == p_sha, 'sidecar P SHA-256 differs from P'
        with np.load(path, allow_pickle=False) as z:
            key = 'img' if kind == 'student' else 'prob'
            assert z.files == [key], z.files
            x = z[key]
        detail['norm_err'] = (vl3_student if kind == 'student' else vl3_own)(x, n_rows)
        assert side['sha256'] == array_sha256(x), 'sidecar array hash differs'
        detail['ok'] = True
    except (AssertionError, OSError, KeyError, ValueError) as e:
        detail.update(ok=False, error=f'{type(e).__name__}: {e}')
    return detail['ok'], detail


# ------------------------------------------------------------------ inference
class Run:
    def __init__(self, args):
        import rb_data
        eg_paths.check_imports()
        eg_encoders.extend_rb_maps()
        self.args = args
        self.outputs = args.outputs
        self.names = rb_data.p_names()
        assert rb_data.names_sha256(self.names) == eg_paths.P_SHA256, 'P differs from 12-24\'s'
        if args.smoke:
            assert eg_paths.smoke_ok(self.outputs), 'smoke needs --outputs under /tmp/claude-0/'
            self.names = self.names[:args.n_rows]
        self.p_sha = rb_data.names_sha256(self.names)
        self.n_rows = len(self.names)
        self.images = {}
        self.commit = code_commit()
        self.base = Path(self.outputs) / STAGE / 'p'
        self.index_path = self.base / 'index.json'

    def store_images(self, store):
        """The encoder's default store rows for P's names (rb_encoders.full_images), float32 unit norm."""
        import rb_encoders
        if store not in self.images:
            x = rb_encoders.full_images(store, self.names)
            assert x.shape == (self.n_rows, eg_encoders.BY_STORE[store].width)
            self.images[store] = x
        return self.images[store]

    def vl4(self, arms, missing):
        """VL4: with an index or any export present, only the same plan may continue (resume); nothing is added."""
        exports = sorted(self.base.rglob('*.npz')) if self.base.exists() else []
        if not self.index_path.exists():
            assert not exports, f'VL4: exports exist without {self.index_path}; stage ii stops'
            return
        index = json.loads(self.index_path.read_text())
        same = (index['arms'], index['missing'], index['seeds'], index['aspects'], index['p_sha256']) == \
            (arms, missing, list(self.seeds), list(self.aspects), self.p_sha)
        assert same, 'VL4: an arm or seed would be added, dropped or changed after the plan was written ' \
                     f'({self.index_path}); stage ii stops'

    def student_one(self, enc, aspect, seed, forced):
        import rb_student
        out = eg_paths.p_student(self.outputs, enc.student, aspect, seed)
        if out.exists():
            ok, detail = check_export(out, self.p_sha, self.n_rows, 'student')
            if ok:
                return 'kept'
            assert forced, f'VL4: {out} fails VL3 ({detail["error"]}); a rerun needs --forced-vl3 CAUSE'
        weights = rb_student.weights_path(weights_root(self.outputs, enc.student), enc.student, aspect, seed)
        info = json.loads(weights.with_suffix('.json').read_text())
        w_sha = rb_student.ms_train.head.sha_file(weights)
        assert w_sha == info['weights_sha256'], f'{weights}: weights checksum mismatch'
        self.prepare_rerun(out, w_sha, forced, detail if out.exists() else None)
        img = rb_student.infer(weights, aspect, enc.width, self.store_images(enc.store), self.args.device)
        vl3_student(img, self.n_rows)
        _atomic_npz(out, img=img)
        _atomic_json(out.with_suffix('.json'), dict(
            arm=enc.student, aspect=aspect, seed=seed, store=enc.store, batch=BATCH, p_sha256=self.p_sha,
            weights_sha256=w_sha, weights=str(weights), rows=self.n_rows, width=int(img.shape[1]), dtype='float16',
            sha256=array_sha256(img),
            device=self.args.device, smoke=bool(self.args.smoke), code_commit=self.commit))
        return 'wrote'

    def own_one(self, enc, seed, forced):
        import rb_matched
        out = eg_paths.p_own(self.outputs, enc.own, seed)
        if out.exists():
            ok, detail = check_export(out, self.p_sha, self.n_rows, 'own')
            if ok:
                return 'kept'
            assert forced, f'VL4: {out} fails VL3 ({detail["error"]}); a rerun needs --forced-vl3 CAUSE'
        ck = own_checkpoint(self.outputs, enc.own, seed)
        meta = rb_matched.base.verify_output(ck)
        device = meta['config']['device']       # rb_matched.infer insists on the training device
        if str(device).startswith('cuda') and not __import__('torch').cuda.is_available():
            assert self.args.smoke, f'{enc.own}: trained on {device}, which is not available'
            return 'skipped (smoke without GPU)'
        self.prepare_rerun(out, meta['weights_sha256'], forced, detail if out.exists() else None)
        prob = rb_matched.infer(ck, self.store_images(enc.store), device)
        vl3_own(prob, self.n_rows)
        _atomic_npz(out, prob=prob)
        _atomic_json(out.with_suffix('.json'), dict(
            arm=enc.own, seed=seed, store=enc.store, batch='one batch (rb_matched.infer)', p_sha256=self.p_sha,
            weights_sha256=meta['weights_sha256'], weights=str(ck), rows=self.n_rows, width=int(prob.shape[1]),
            dtype='float32',
            sha256=array_sha256(prob), device=str(device), smoke=bool(self.args.smoke), code_commit=self.commit))
        return 'wrote'

    def prepare_rerun(self, out, weights_sha256, forced, detail=None):
        """Before replacing an export, enforce VL4's same weights/code and record the rerun."""
        side = out.with_suffix('.json')
        complete = self.index_path.exists() and json.loads(self.index_path.read_text()).get('complete')
        if not out.exists() and not side.exists() and not complete:
            return
        assert side.is_file(), f'VL4: {out}: cannot verify same weights and code without the old sidecar'
        old = json.loads(side.read_text())
        assert (old.get('weights_sha256'), old.get('code_commit')) == (weights_sha256, self.commit), \
            f'VL4: {out}: a rerun requires the same weights and code as the old sidecar'
        if out.exists():
            self.log_forced(out, detail, forced)
        else:
            self.log_forced(out, dict(error='export missing'), forced or 'export missing; re-inferred from saved weights')

    def log_forced(self, out, detail, cause):
        line = dict(file=str(out), cause=cause, failed=detail, code_commit=self.commit)
        log = Path(self.outputs) / STAGE / 'vl4_log.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as f:
            f.write(json.dumps(line, allow_nan=False) + '\n')

    def infer(self):
        args = self.args
        self.aspects, self.seeds = eg_paths.ASPECTS, eg_paths.SEEDS
        encoders = eg_encoders.ENCODERS
        if args.arms or args.seeds is not None:
            assert args.smoke, 'VL4: --arms and --seeds are only for a smoke run'
            if args.arms:
                encoders = tuple(e for e in encoders if e.student in args.arms or e.own in args.arms)
            if args.seeds is not None:
                self.seeds = tuple(args.seeds)
        if not args.smoke:
            eg_run.require_stage_ii_ready(self.outputs)
        arms, missing = plan(self.outputs, self.aspects, self.seeds, encoders)
        if args.arms:
            arms = [a for a in arms if a in args.arms]
            missing = {a: r for a, r in missing.items() if a in args.arms}
        self.vl4(arms, missing)
        index = dict(arms=arms, missing=missing, seeds=list(self.seeds), aspects=list(self.aspects),
                     p_sha256=self.p_sha, n_rows=self.n_rows, complete=False, smoke=bool(args.smoke),
                     code_commit=self.commit)
        if not self.index_path.exists():
            _atomic_json(self.index_path, index)
        for enc in encoders:
            if enc.student in arms:
                done = [self.student_one(enc, a, s, args.forced_vl3) for s in self.seeds for a in self.aspects]
                print(f'{enc.student}: {len(done)} files, {done.count("wrote")} written', flush=True)
            if enc.own in arms:
                done = [self.own_one(enc, s, args.forced_vl3) for s in self.seeds]
                print(f'{enc.own}: {len(done)} files, {done.count("wrote")} written ({done[0]})', flush=True)
        for arm, reason in missing.items():
            print(f'{arm}: missing ({reason})', flush=True)
        _atomic_json(self.index_path, dict(index, complete=True))

    # -------------------------------------------------------------- VL1 (b), (c), VL3
    def check(self):
        import rb_encoders
        import rb_matched
        import rb_student
        index = json.loads(self.index_path.read_text())
        assert index['complete'], 'the P inference did not finish'
        assert index['p_sha256'] == self.p_sha
        vl3 = {}
        for arm in index['arms']:
            enc = eg_encoders.BY_STUDENT.get(arm) or eg_encoders.BY_OWN[arm]
            if arm == enc.student:
                files = [(eg_paths.p_student(self.outputs, arm, a, s), 'student')
                         for s in index['seeds'] for a in index['aspects']]
            else:
                files = [(eg_paths.p_own(self.outputs, arm, s), 'own') for s in index['seeds']]
            for path, kind in files:
                if kind == 'own' and not path.exists() and self.args.smoke:
                    continue            # a smoke without GPU skips the own arms
                ok, detail = check_export(path, self.p_sha, self.n_rows, kind)
                vl3[str(path.relative_to(self.outputs))] = detail
        if not self.args.smoke:
            assert self.p_sha == eg_paths.P_SHA256 and self.n_rows == N_P
        vl3_ok = bool(vl3) and all(d['ok'] for d in vl3.values())
        # (b) K4 seed 0 equals 12-24's saved R6-5 P images, bit for bit, three aspects
        b = {}
        for a in eg_paths.ASPECTS:
            ref = eg_paths.RB_PART1 / f'train/R6-5/{a}__s0__P.npz'
            with np.load(ref, allow_pickle=False) as z:
                theirs = z['img'][:self.n_rows]
            mine_path = eg_paths.p_student(self.outputs, 'K4', a, 0)
            try:
                with np.load(mine_path, allow_pickle=False) as z:
                    mine = z['img']
                b[a] = dict(equal=bool(mine.dtype == theirs.dtype and mine.shape == theirs.shape
                                       and mine.tobytes() == theirs.tobytes()), ref=str(ref))
            except OSError as e:
                b[a] = dict(equal=False, error=str(e), ref=str(ref))
        b_ok = all(v['equal'] for v in b.values())
        # (c) M2 seed 0 on the held images equals 12-24's saved held output, bit for bit
        held_ref = eg_paths.RB_PART2 / SCALE / 'held/M2/P3__s0.npz'
        ck = own_checkpoint(self.outputs, 'M2', 0)
        meta = rb_matched.base.verify_output(ck)
        device = meta['config']['device']
        if str(device).startswith('cuda') and not __import__('torch').cuda.is_available():
            c = dict(equal=False, error=f'M2 was trained on {device}, which is not available', ref=str(held_ref))
        else:
            held, held_sha = rb_student.held_names()
            assert held_sha == eg_paths.HELD_SHA256
            prob = rb_matched.infer(ck, rb_encoders.full_images('siglip2_so400m', held), device)
            with np.load(held_ref, allow_pickle=False) as z:
                theirs = z['prob']
            c = dict(equal=bool(prob.dtype == theirs.dtype and prob.shape == theirs.shape
                                and prob.tobytes() == theirs.tobytes()), rows=int(len(prob)), device=str(device),
                     ref=str(held_ref))
        record = {'item': 'VL1 (b), (c), VL3', 'pass': bool(b_ok and c['equal'] and vl3_ok), 'b': dict(pass_=b_ok, aspects=b),
                  'c': c, 'vl3': dict(pass_=vl3_ok, files=vl3), 'p_sha256': self.p_sha, 'n_rows': self.n_rows,
                  'smoke': bool(self.args.smoke), 'code_commit': self.commit}
        record['b']['pass'] = record['b'].pop('pass_')
        record['vl3']['pass'] = record['vl3'].pop('pass_')
        record['c']['pass'] = c['equal']
        path = Path(self.outputs) / STAGE / 'vl1.json'
        _atomic_json(path, record)
        print(f"VL1 (b) {record['b']['pass']}, (c) {record['c']['pass']}, VL3 {record['vl3']['pass']} "
              f"({len(vl3)} files): pass {record['pass']} -> {path}", flush=True)
        return record['pass']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('infer', 'check', 'all'))
    parser.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    parser.add_argument('--device', default='cuda', help='students; own arms use their checkpoint\'s training device')
    parser.add_argument('--smoke', action='store_true', help='cut P (--n-rows), outputs under /tmp/claude-0/')
    parser.add_argument('--n-rows', type=int, default=1024, help='smoke: first rows of P')
    parser.add_argument('--arms', nargs='*', help='smoke only: arm IDs')
    parser.add_argument('--seeds', nargs='*', type=int, help='smoke only')
    parser.add_argument('--forced-vl3', metavar='CAUSE', help='VL4: rerun only the files that failed VL3')
    args = parser.parse_args(argv)
    for frozen in (eg_paths.RB_OUT, eg_paths.CT_OUT):
        assert not args.outputs.resolve().is_relative_to(frozen.resolve()), f'--outputs is inside the frozen {frozen}'
    run = Run(args)
    if args.command in ('infer', 'all'):
        run.infer()
    if args.command in ('check', 'all'):
        run.aspects, run.seeds = eg_paths.ASPECTS, eg_paths.SEEDS
        return 0 if run.check() else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
