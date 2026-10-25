"""Stage ii scorer (rule §3.2, §4.2): lift of each encoder's student over its own PercepT arm, interval, L outcome."""
import argparse
import hashlib
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

import eg_encoders
import eg_paths  # noqa: F401  (sys.path)
import rb_ami

N_SEEDS = 5
SEED_VALUES = [0, 1, 2, 3, 4]


def outcome(lift, lower):
    """L-HOLDS: lift > 0 and lower bound > 0. L-QUAL: lift > 0, lower bound <= 0. L-FAILS: lift <= 0."""
    assert lift == lift and lower == lower, 'nan lift or bound'
    if lift <= 0:
        return 'L-FAILS'
    return 'L-HOLDS' if lower > 0 else 'L-QUAL'


# ------------------------------------------------------------------ the score A_a(arm, I, S)
def arm_score(y, idx, arm, seeds):
    """A(arm, I, S): the mean over the seeds in S (repeats count) of the mean over that seed's partitions of
    AMI(labels[I], partition[I]). `y` = labels[idx]; `arm[s]` = int [m, n] partitions of seed s over all of P."""
    cache = {}
    for s in seeds:
        s = int(s)
        if s not in cache:
            cache[s] = float(np.mean([ami(y, p[idx]) for p in arm[s]]))
    return float(np.mean([cache[int(s)] for s in seeds]))


# ------------------------------------------------------------------ the draws (rule §3.2, in this order)
def draw_plan(n_labelled, n_draws=rb_ami.N_DRAWS):
    """One `default_rng(0)`; draw b: positions in L_a (`integers(0, n, size=n)`), then S_b for the student, then S_b for
    the own arm (`choice([0..4], size=5, replace=True)` each)."""
    assert n_labelled > 0 and n_draws > 0
    rng = np.random.default_rng(0)
    pos = np.empty((n_draws, n_labelled), dtype=np.int32)
    s_k = np.empty((n_draws, N_SEEDS), dtype=np.int64)
    s_o = np.empty((n_draws, N_SEEDS), dtype=np.int64)
    for b in range(n_draws):
        pos[b] = rng.integers(0, n_labelled, size=n_labelled)
        s_k[b] = rng.choice(SEED_VALUES, size=N_SEEDS, replace=True)
        s_o[b] = rng.choice(SEED_VALUES, size=N_SEEDS, replace=True)
    return dict(pos=pos, seeds_student=s_k, seeds_own=s_o)


_STATE = {}


def _init_worker(state):
    _STATE.update(state)
    threadpool_limits(limits=1)


def _one_draw(b):
    s = _STATE
    idx = s['labelled'][s['plan']['pos'][b]]
    y = s['labels'][idx]
    a_k = arm_score(y, idx, s['student'], s['plan']['seeds_student'][b])
    a_r = arm_score(y, idx, s['own_r20'], s['plan']['seeds_own'][b])
    a_o = arm_score(y, idx, s['own_own'], s['plan']['seeds_own'][b])
    return a_k, a_r, a_o


def bootstrap(labels, student, own_r20, own_own, n_draws=rb_ami.N_DRAWS, workers=8):
    """The per-draw scores of one encoder and aspect: {student, own_r20, own_own, lift} float64 [n_draws].

    lift_b = A(Kk) - max(A(Ok, r20), A(Ok, own)), the maximum taken again in every draw. The result does not depend
    on `workers` (each draw is computed alone)."""
    labels = np.asarray(labels)
    labelled = np.flatnonzero(labels >= 0)
    assert 1 <= workers <= 8 and len(student) == len(own_r20) == len(own_own) == N_SEEDS
    for arm in (student, own_r20, own_own):
        assert all(np.shape(p)[1:] == (len(labels),) and np.ndim(p) == 2 for p in arm), 'partition shapes'
    state = dict(labels=labels, labelled=labelled, plan=draw_plan(len(labelled), n_draws), student=student,
                 own_r20=own_r20, own_own=own_own)
    if workers == 1 or n_draws < 2:
        _STATE.update(state)
        rows = [_one_draw(b) for b in range(n_draws)]
    else:
        with multiprocessing.get_context('fork').Pool(workers, initializer=_init_worker, initargs=(state,)) as pool:
            rows = pool.map(_one_draw, range(n_draws), chunksize=max(1, n_draws // (workers * 8)))
    rows = np.asarray(rows, dtype=np.float64)
    assert rows.shape == (n_draws, 3) and np.isfinite(rows).all()
    return dict(student=rows[:, 0], own_r20=rows[:, 1], own_own=rows[:, 2],
                lift=rows[:, 0] - np.maximum(rows[:, 1], rows[:, 2]))


# ------------------------------------------------------------------ one encoder, one aspect
def score_aspect(labels, student, own_r20, own_own, frozen, n_draws=rb_ami.N_DRAWS, workers=8):
    """Point scores on all of L_a with S = {0..4}, the lift and its 95% interval (the outcome is the encoder level's).

    `frozen` = int [3, n] partitions of the frozen image (no seed)."""
    labels = np.asarray(labels)
    labelled = np.flatnonzero(labels >= 0)
    y = labels[labelled]
    assert np.shape(frozen) == (3, len(labels)), np.shape(frozen)
    a = dict(student=arm_score(y, labelled, student, SEED_VALUES), own_r20=arm_score(y, labelled, own_r20, SEED_VALUES),
             own_own=arm_score(y, labelled, own_own, SEED_VALUES), frozen=arm_score(y, labelled, [frozen], [0]))
    best = 'own_r20' if a['own_r20'] >= a['own_own'] else 'own_own'
    lift = a['student'] - a[best]
    boot = bootstrap(labels, student, own_r20, own_own, n_draws, workers)
    lo, hi = np.percentile(boot['lift'], [2.5, 97.5])
    return dict(n_labelled=int(len(labelled)), n_draws=int(n_draws), A=a, own_best=best, lift=float(lift),
                interval=[float(lo), float(hi)], student_minus_frozen=float(a['student'] - a['frozen']))


# ------------------------------------------------------------------ the exports (ticket 07's layout)
N_P = 29232
WIDTH = 2048                        # students' projected image width
N_TOPICS = 40
STAGE = 'stage_ii'


def array_sha256(x):
    """12-24's array hash (`rb_pick.array_sha256`): SHA-256 of the contiguous bytes."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def code_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=eg_paths.ROUND, capture_output=True, text=True,
                          check=True).stdout.strip()


def rb_ami_sha256():
    return hashlib.sha256(Path(rb_ami.__file__).read_bytes()).hexdigest()


def versions():
    v = rb_ami.versions()
    return dict(sklearn=v['sklearn'], numpy=v['numpy'])


def require_vl1(outputs):
    """The scorer states no outcome until VL1 holds: `vl1.json` (b, c, VL3; ticket 07) and `vl1_scorer.json` (a)."""
    for name in ('vl1.json', 'vl1_scorer.json'):
        path = Path(outputs) / STAGE / name
        assert path.is_file(), f'VL1: {path} missing; no outcome before VL1 holds'
        record = json.loads(path.read_text())
        assert record.get('pass') is True, f'VL1: {path} does not pass; stage ii stops (rule §4.2)'


def read_missing(outputs):
    """{arm id: reason} of the encoders without stage 3 weights, from `stage_ii/p/index.json` key `missing`
    (a dict arm -> reason, or a list of arms). The arm may be the student, own arm, frozen image or store name."""
    index = json.loads((Path(outputs) / STAGE / 'p' / 'index.json').read_text())
    missing = index.get('missing', {})
    if isinstance(missing, list):
        missing = {name: 'listed as missing in stage_ii/p/index.json' for name in missing}
    assert isinstance(missing, dict) and all(isinstance(v, str) for v in missing.values()), 'index.json: missing'
    return missing


def _check_sidecar(path, p_sha):
    side = path.with_suffix('.json')
    assert side.is_file(), f'{side}: sidecar missing'
    record = json.loads(side.read_text())
    assert record.get('p_sha256') == p_sha, f'{side}: P sha256 differs from P'
    return record


def load_student(outputs, arm, aspect, seed, n_rows=N_P, width=WIDTH, p_sha=eg_paths.P_SHA256):
    """(img float16 [n, width], its SHA-256) of a student's P export after the sidecar and VL3 checks."""
    path = eg_paths.p_student(outputs, arm, aspect, seed)
    _check_sidecar(path, p_sha)
    with np.load(path, allow_pickle=False) as z:
        assert 'img' in z.files, (path, z.files)
        img = z['img']
    assert img.dtype == np.float16 and img.shape == (n_rows, width), f'{path}: {img.dtype} {img.shape}'
    rb_ami.unit_float32(img)                      # finite and unit norm within 1e-3
    return img, array_sha256(img)


def load_own(outputs, arm, seed, n_rows=N_P, p_sha=eg_paths.P_SHA256):
    """(prob float32 [n, 40], its SHA-256) of an own arm's P export after the sidecar checks."""
    path = eg_paths.p_own(outputs, arm, seed)
    _check_sidecar(path, p_sha)
    with np.load(path, allow_pickle=False) as z:
        assert 'prob' in z.files, (path, z.files)
        prob = z['prob']
    assert prob.dtype == np.float32 and prob.shape == (n_rows, N_TOPICS), f'{path}: {prob.dtype} {prob.shape}'
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, f'{path}: not sigmoid probabilities'
    assert (np.linalg.norm(prob, axis=1) > 0).all(), f'{path}: a zero row'
    return prob, array_sha256(prob)


def row_normalized(prob):
    """The 40 sigmoid values divided by their row norm (float32), the own arm's vector reading."""
    return (prob / np.linalg.norm(prob, axis=1, keepdims=True)).astype(np.float32)


def argmax_partition(prob):
    """The own arm's "own" reading: the arg-max topic, one partition, int64 [1, n]."""
    return np.argmax(prob, axis=1).astype(np.int64)[None, :]


# ------------------------------------------------------------------ fixed partitions
def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def cached_partitions(folder, arm, name, kind, source_sha, p_sha, compute):
    """int64 [m, n] partitions saved once as `<folder>/<arm>/<name>.npz` (key `labels`) with a `.json` identity; a saved
    set is reused only if its inputs (source array, P, versions, rb_ami.py) are the same, else the run stops."""
    path = Path(folder) / arm / f'{name}.npz'
    meta_path = path.with_suffix('.json')
    identity = dict(arm=arm, name=name, kind=kind, p_sha256=p_sha, source_sha256=source_sha, **versions(),
                    rb_ami_sha256=rb_ami_sha256())
    if path.exists() or meta_path.exists():
        assert path.exists() and meta_path.exists(), f'{path}: partial partition file'
        meta = json.loads(meta_path.read_text())
        assert meta['identity'] == identity, f'{path}: saved partitions came from other inputs; move them aside'
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['labels']
            labels = z['labels']
        assert array_sha256(labels) == meta['sha256'], f'{path}: sha256 differs from its record'
    else:
        labels = compute()
        assert labels.dtype == np.int64 and labels.ndim == 2, (path, labels.dtype, labels.shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.stem + '.pending.npz')
        np.savez(pending, labels=labels)
        pending.replace(path)
        _atomic_json(meta_path, dict(identity=identity, sha256=array_sha256(labels), shape=list(labels.shape),
                                     k=rb_ami.K, states=list(rb_ami.STATES), threads=rb_ami.THREADS))
    assert labels.dtype == np.int64 and labels.ndim == 2, (path, labels.dtype, labels.shape)
    return labels


# ------------------------------------------------------------------ the run
def run(outputs, labels, *, frozen_images, p_sha=eg_paths.P_SHA256, n_rows=N_P, width=WIDTH, n_draws=rb_ami.N_DRAWS,
        workers=8, encoders=eg_encoders.ENCODERS, meta=None):
    """Score every stage ii encoder and write `stage_ii/lift.json`; returns the record.

    `labels` = {aspect: int64 [n_rows]} in P order (-1 masked). `frozen_images(store)` = float32 [n_rows, w], unit norm.
    Refuses unless VL1 holds. An encoder listed missing in `stage_ii/p/index.json` is reported missing, no outcome."""
    outputs = Path(outputs)
    require_vl1(outputs)
    assert set(labels) == set(eg_paths.ASPECTS)
    for a in eg_paths.ASPECTS:
        assert labels[a].dtype == np.int64 and labels[a].shape == (n_rows,) and labels[a].min() >= -1, a
        assert (labels[a] >= 0).any(), a
    missing = read_missing(outputs)
    folder = outputs / STAGE / 'partitions'
    record, sources = {}, {}
    for enc in encoders:
        reasons = [missing[n] for n in (enc.student, enc.own, enc.frozen, enc.store) if n in missing]
        if reasons:
            record[enc.student] = dict(encoder=enc.store, student=enc.student, own=enc.own, frozen=enc.frozen,
                                       status='missing', reason=reasons[0], outcome=None)
            continue
        src = sources[enc.student] = {}
        student = {a: [] for a in eg_paths.ASPECTS}
        own_r20, own_own = [], []
        for seed in eg_paths.SEEDS:
            for a in eg_paths.ASPECTS:
                img, sha = load_student(outputs, enc.student, a, seed, n_rows, width, p_sha)
                src[f'{enc.student}/{a}__s{seed}'] = sha
                student[a].append(cached_partitions(folder, enc.student, f'{a}__s{seed}', 'r20', sha, p_sha,
                                                    lambda img=img: rb_ami.partitions(img)))
            prob, sha = load_own(outputs, enc.own, seed, n_rows, p_sha)
            src[f'{enc.own}/P3__s{seed}'] = sha
            own_r20.append(cached_partitions(folder, enc.own, f'P3__s{seed}__r20', 'r20', sha, p_sha,
                                             lambda prob=prob: rb_ami.partitions(row_normalized(prob))))
            own_own.append(cached_partitions(folder, enc.own, f'P3__s{seed}__own', 'own', sha, p_sha,
                                             lambda prob=prob: argmax_partition(prob)))
        frozen = np.asarray(frozen_images(enc.store))
        assert frozen.dtype == np.float32 and frozen.ndim == 2 and frozen.shape[0] == n_rows, (frozen.dtype, frozen.shape)
        src[enc.frozen] = array_sha256(frozen)
        frozen_parts = cached_partitions(folder, enc.frozen, 'default', 'r20', src[enc.frozen], p_sha,
                                         lambda: rb_ami.partitions(frozen))
        aspects = {a: score_aspect(labels[a], student[a], own_r20, own_own, frozen_parts, n_draws, workers)
                   for a in eg_paths.ASPECTS}
        emotion = aspects['emotion']
        record[enc.student] = dict(encoder=enc.store, student=enc.student, own=enc.own, frozen=enc.frozen,
                                   status='scored', aspects=aspects,
                                   outcome=outcome(emotion['lift'], emotion['interval'][0]))
        print(f'{enc.student}: {record[enc.student]["outcome"]}', flush=True)
    result = dict(p_sha256=p_sha, n_rows=n_rows, n_draws=n_draws,
                  label_sha256={a: array_sha256(labels[a]) for a in eg_paths.ASPECTS},
                  label_counts={a: int((labels[a] >= 0).sum()) for a in eg_paths.ASPECTS}, versions=versions(),
                  threads=rb_ami.THREADS, code_commit=code_commit(), rb_ami_sha256=rb_ami_sha256(), sources=sources,
                  encoders=record, **(meta or {}))
    result = json.loads(json.dumps(result, allow_nan=False))
    path = outputs / STAGE / 'lift.json'
    if path.exists():
        old = json.loads(path.read_text())
        drop = lambda d: {k: v for k, v in d.items() if k != 'code_commit'}
        assert drop(old) == drop(result), f'{path} exists with other results; move it aside'
    else:
        _atomic_json(path, result)
    return result


# ------------------------------------------------------------------ VL1 (a): this scorer on 12-24's part 1 exports
VL1A_TOL = 1e-6
VL1A_RECIPES = ('R0-5', 'R6-5')
VL1A_REFERENCES = ('default', 'emotion_focus', 'style_word', 'genre_word')


def point_a(labels, parts):
    """A on all of L_a for one partition set: the scorer's point estimate (S = {0}, one seed, all rows labelled)."""
    labels = np.asarray(labels)
    labelled = np.flatnonzero(labels >= 0)
    return arm_score(labels[labelled], labelled, [parts], [0])


def check_vl1a(outputs, labels, *, load_export, references, pick, reported, p_sha=eg_paths.P_SHA256, tol=VL1A_TOL):
    """VL1 (a): this scorer's partitions and A on 12-24's saved part 1 exports and Qwen images give `pick.json`'s A of
    R0-5 and R6-5 (`pick['recipes'][r][a]['A']`) and `reported.json`'s `references_p` values, within `tol`.

    `load_export(recipe, aspect)` = float16 img of the part 1 export; `references` = {key: callable -> float32 [n, 2048]}.
    Caches go under `outputs/stage_ii/partitions/vl1a`. Writes `stage_ii/vl1_scorer.json` (also when it fails)."""
    outputs = Path(outputs)
    folder = outputs / STAGE / 'partitions'
    items = []

    def add(name, computed, expected):
        items.append(dict(name=name, computed=float(computed), expected=float(expected),
                          diff=float(abs(computed - expected)), ok=bool(abs(computed - expected) <= tol)))

    for recipe in VL1A_RECIPES:
        for a in eg_paths.ASPECTS:
            img = load_export(recipe, a)
            parts = cached_partitions(folder, 'vl1a', f'{recipe}__{a}', 'r20', array_sha256(img), p_sha,
                                      lambda img=img: rb_ami.partitions(img))
            add(f'pick/{recipe}/{a}', point_a(labels[a], parts), pick['recipes'][recipe][a]['A'])
    assert set(references) == set(VL1A_REFERENCES), sorted(references)
    for key in VL1A_REFERENCES:
        x = np.asarray(references[key]())
        parts = cached_partitions(folder, 'vl1a', f'ref_{key}', 'r20', array_sha256(x), p_sha,
                                  lambda x=x: rb_ami.partitions(x))     # one k-means set serves all three aspects
        for a in eg_paths.ASPECTS:
            add(f'references_p/{key}/{a}', point_a(labels[a], parts), reported['references_p'][key][a])
    record = dict(item='VL1 (a)', tol=tol, items=items, max_abs_diff=max(i['diff'] for i in items),
                  **{'pass': bool(all(i['ok'] for i in items))}, p_sha256=p_sha, versions=versions(),
                  code_commit=code_commit(), rb_ami_sha256=rb_ami_sha256())
    record = json.loads(json.dumps(record, allow_nan=False))
    path = outputs / STAGE / 'vl1_scorer.json'
    if path.exists():
        drop = lambda d: {k: v for k, v in d.items() if k != 'code_commit'}
        assert drop(json.loads(path.read_text())) == drop(record), f'{path} exists with other results; move it aside'
    else:
        _atomic_json(path, record)
    return record


# ------------------------------------------------------------------ command line
def _inputs(args):
    """P names and labels from `rb_data`, checked against 12-24's P and its label hashes."""
    import rb_data
    import rb_pick
    outputs = args.outputs.resolve()
    for frozen in (eg_paths.RB_OUT, eg_paths.CT_OUT):
        assert not outputs.is_relative_to(frozen.resolve()), f'--outputs {outputs} is inside the frozen {frozen}'
    if args.smoke:
        assert eg_paths.smoke_ok(args.outputs), 'smoke needs --outputs under /tmp/claude-0/'
    eg_paths.check_imports()
    names = rb_data.p_names()
    assert rb_data.names_sha256(names) == eg_paths.P_SHA256, 'P differs from 12-24\'s'
    labels = rb_data.p_labels()
    pick = json.loads((eg_paths.RB_PART1 / 'pick.json').read_text())
    assert pick['p_sha256'] == eg_paths.P_SHA256
    for a in eg_paths.ASPECTS:
        assert rb_pick.array_sha256(labels[a]) == pick['label_sha256'][a], f'{a}: label hash differs from the pick\'s'
        assert array_sha256(labels[a]) == pick['label_sha256'][a]
    return names, labels, pick


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage ii scorer: lifts, intervals, L outcomes; VL1 (a).')
    parser.add_argument('command', choices=('score', 'check'),
                        help='score: stage_ii/lift.json (needs stage_ii/vl1.json and vl1_scorer.json to pass); '
                             'check: VL1 (a) on 12-24\'s part 1 exports, writes stage_ii/vl1_scorer.json')
    parser.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true', help='score only: 20 draws; outputs under /tmp/claude-0/')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args(argv)
    names, labels, pick = _inputs(args)
    import rb_data
    import rb_pick
    if args.command == 'check':
        reported = json.loads((eg_paths.RB_PART1 / 'reported.json').read_text())
        assert reported['p_sha256'] == eg_paths.P_SHA256
        references = dict(default=rb_data.p_default_images,
                          **{k: (lambda k=k: rb_data.p_instructed_images(k)) for k in rb_data.INSTRUCTED_KEYS})
        record = check_vl1a(args.outputs, labels, load_export=lambda r, a: rb_pick.load_export(eg_paths.RB_OUT, r, a, eg_paths.P_SHA256),
                            references=references, pick=pick, reported=reported)
        print(f"VL1 (a): pass {record['pass']}, max |diff| {record['max_abs_diff']:.3g} over {len(record['items'])} values "
              f"(tol {record['tol']})", flush=True)
        return 0 if record['pass'] else 1
    import rb_encoders
    eg_encoders.extend_rb_maps()
    result = run(args.outputs, labels, frozen_images=lambda store: rb_encoders.full_images(store, names),
                 n_draws=20 if args.smoke else rb_ami.N_DRAWS, workers=args.workers, meta=dict(smoke=bool(args.smoke)))
    print(f"{args.outputs / STAGE / 'lift.json'}: " + ', '.join(f"{k} {v['outcome']}" for k, v in result['encoders'].items()),
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
