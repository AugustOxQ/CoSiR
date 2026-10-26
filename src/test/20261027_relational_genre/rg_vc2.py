"""VC2: independent stage C calculation from RG §3.2/4.3, RB §3.2 and CT §3.

No held scorer is imported. Only rb_v2's low-level partition/draw helpers are
used; the masked AMIs, two statistics and decisions are derived here.

  python rg_vc2.py --outputs DIR --arm GR-8 [--workers 8] [--smoke]
  python rg_vc2.py --outputs /tmp/claude-0/vc2-k4 --check-k4 --workers 8
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

import rg_paths
from rb_v2 import draw_plan, r20_partition, sha256_names

SEEDS = tuple(range(5))
TOL = 1e-6
_BOOT = {}


def decide(d_t, lower_t, d_r, lower_r, half_gap):
    parts = dict(d_R_ge_G_half=bool(d_r >= half_gap), lower_gt_0=bool(lower_r > 0))
    pc = all(parts.values())
    if lower_t > -.0128:
        outcome = 'G-GO'
    elif d_t >= -.0128:
        outcome = 'G-MM'
    elif pc:
        outcome = 'G-PART'
    else:
        outcome = 'G-NO'
    return dict(outcome_id=outcome, pc=pc, pc_parts=parts,
                matches_or_exceeds=bool(lower_t > 0),
                partial_sentence=bool(pc and outcome in ('G-MM', 'G-PART')))


def _scores(labels, parts, rows):
    rows = rows[labels[rows] >= 0]
    return {s: float(np.mean([adjusted_mutual_info_score(labels[rows], p[rows]) for p in ps]))
            for s, ps in parts.items()}


def _bootstrap_draw(b):
    st = _BOOT
    rows, seeds = st['rows'][b], st['seeds'][b]
    with threadpool_limits(limits=1):
        g = _scores(st['labels'], {s: st['g'][s] for s in set(seeds['G'])}, rows)
        left = float(np.mean([g[s] for s in seeds['G']]))
        if st['stat'] == 'T':
            right = _scores(st['labels'], {0: st['other']}, rows)[0]
        else:
            r = _scores(st['labels'], {s: st['other'][s] for s in set(seeds['R'])}, rows)
            right = float(np.mean([r[s] for s in seeds['R']]))
    return left - right


def calculate(labels, g, t, r, draws=2000, workers=8):
    """Two independently seeded paired bootstraps, masking labels after row draws.

    g/r map model seeds 0..4 to int partitions [3,n]; t is [3,n].
    The public seam also accepts small synthetic held sets.
    """
    labels = np.asarray(labels)
    n = len(labels)
    assert labels.shape == (n,) and labels.dtype.kind in 'iu' and (labels >= 0).any()
    assert sorted(g) == sorted(r) == list(SEEDS)
    assert draws > 0 and 1 <= workers <= 8
    for ps in [t, *g.values(), *r.values()]:
        assert ps.shape == (3, n) and ps.dtype.kind in 'iu' and (ps >= 0).all()
    initial = np.arange(n)
    pg, pt, pr = _scores(labels, g, initial), _scores(labels, {0: t}, initial)[0], _scores(labels, r, initial)
    ag, ar = float(np.mean(list(pg.values()))), float(np.mean(list(pr.values())))
    result = dict(X=.0128, G_half=(pt - ar) / 2, labelled_rows=int((labels >= 0).sum()))
    for stat, other, point, per in [('T', t, pt, pt), ('R', r, ar, {str(s): pr[s] for s in SEEDS})]:
        pools = {'G': SEEDS} if stat == 'T' else {'G': SEEDS, 'R': SEEDS}
        rows, seeds = draw_plan(n, pools, draws)
        _BOOT.update(labels=labels, g=g, other=other, rows=rows, seeds=seeds, stat=stat)
        try:
            if workers == 1:
                boot = list(map(_bootstrap_draw, range(draws)))
            else:
                with mp.get_context('fork').Pool(workers) as pool:
                    boot = pool.map(_bootstrap_draw, range(draws), chunksize=max(1, draws // (workers * 4)))
        finally:
            _BOOT.clear()
        result[f'd_{stat}'] = dict(A_G=ag, **{f'A_{stat}': point}, d=ag - point,
                                   interval=np.percentile(boot, [2.5, 97.5]).tolist(), bootstrap_d=boot,
                                   per_seed={'G': {str(s): pg[s] for s in SEEDS}, stat: per})
    result.update(decide(result['d_T']['d'], result['d_T']['interval'][0],
                         result['d_R']['d'], result['d_R']['interval'][0], result['G_half']))
    result['bootstrap'] = dict(resamples=draws, rng='numpy.random.default_rng(0)', d_T_draws='I, G',
                               d_R_draws='I, G, R', n_rows=n, percentiles=[2.5, 97.5], method='linear')
    return result


def _numeric_differences(mine, theirs, prefix, differences, deltas):
    """Record every differing scalar, including each changed bootstrap draw."""
    if isinstance(mine, dict):
        if not isinstance(theirs, dict):
            differences.append(f'{prefix}: missing or malformed mapping')
            return
        for key, value in mine.items():
            _numeric_differences(value, theirs.get(key), f'{prefix}.{key}', differences, deltas)
    elif isinstance(mine, list):
        if not isinstance(theirs, list) or len(mine) != len(theirs):
            differences.append(f'{prefix}: sequence length differs or is missing')
            return
        for i, (a, b) in enumerate(zip(mine, theirs)):
            _numeric_differences(a, b, f'{prefix}[{i}]', differences, deltas)
    else:
        if not isinstance(theirs, (int, float)) or isinstance(theirs, bool):
            differences.append(f'{prefix}: missing or nonnumeric value {theirs!r}')
            return
        delta = abs(mine - theirs)
        deltas.append(delta)
        if not delta <= TOL:
            differences.append(f'{prefix}: re-derived {mine!r}, reported {theirs!r}, abs difference {delta!r}')


def compare_verdict(mine, verdict):
    """Compare only after derive; report all scalar/interval/draw differences."""
    differences = list(mine['partitions']['differences'])
    deltas = []
    for key in ('X', 'G_half', 'G_half_frozen', 'labelled_rows'):
        _numeric_differences(mine[key], verdict.get(key), key, differences, deltas)
    _numeric_differences(mine['G_half'], mine['G_half_frozen'], 'frozen half gap', differences, deltas)
    for stat in ('d_T', 'd_R'):
        keys = ('A_G', 'A_T' if stat == 'd_T' else 'A_R', 'd', 'interval', 'bootstrap_d')
        _numeric_differences({k: mine[stat][k] for k in keys}, verdict.get(stat), stat, differences, deltas)
    for key in ('outcome_id', 'matches_or_exceeds', 'pc', 'pc_parts', 'partial_sentence',
                'arm', 'held_sha256', 'rule_commit', 'smoke'):
        if type(verdict.get(key)) is not type(mine[key]) or mine[key] != verdict[key]:
            differences.append(f'{key}: re-derived {mine[key]!r}, reported {verdict.get(key)!r}')
    # Each statistic must have the requested number of resamples, including smoke.
    for key in ('resamples', 'n_rows', 'percentiles'):
        _numeric_differences(mine['bootstrap'][key], verdict.get('bootstrap', {}).get(key),
                             f'bootstrap.{key}', differences, deltas)
    return dict(agree=not differences, differences=differences,
                max_abs_diff=float(max(deltas, default=0)), tolerance=TOL)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Inputs:
    held: Path = rg_paths.CT_OUT / 'held/held_paintings.json'
    annotations: Path = Path('/data/PDD/artelingo/artelingo_train.json')
    genre_csvs: tuple = (Path('/data/SSD/wikiart_genre/genre_train.csv'),
                        Path('/data/SSD/wikiart_genre/genre_val.csv'))
    teacher: Path = rg_paths.PRE / 'artelingo_full/qwen3vl_emb_2b_aspect_prompts/genre_word_held'
    rb: Path = rg_paths.RB_PART2
    held_sha: str = rg_paths.HELD_SHA256
    n: int = rg_paths.N_HELD
    width: int = rg_paths.WIDTH
    labelled: int = 9949


def genre_labels(names, annotations, genre_csvs):
    """First caption in annotation order, joined by painting to the WikiArt IDs.

    Genre is a painting attribute in these annotations, supplied by the two
    WikiArt tables; missing or conflicting table entries receive -1.
    """
    codes = {}
    for path in genre_csvs:
        with Path(path).open() as handle:
            for relative, raw in csv.reader(handle):
                name, code = Path(relative).stem, int(raw)
                assert 0 <= code < 10, (path, code)
                if name in codes and codes[name] != code:
                    codes[name] = -1
                else:
                    codes.setdefault(name, code)
    first = {}
    rows = json.loads(Path(annotations).read_text())
    assert isinstance(rows, list) and rows
    for row in rows:
        assert isinstance(row['painting'], str)
        first.setdefault(row['painting'], codes.get(row['painting'], -1))
    assert all(name in first for name in names), 'held painting without captions'
    return np.asarray([first[name] for name in names], dtype=np.int64)


def _unit_array(x, n, width, path):
    assert x.shape == (n, width) and x.dtype == np.float16, (path, x.shape, x.dtype)
    assert np.isfinite(x).all(), (path, 'nonfinite output')
    assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < 1e-3, (path, 'unit norm')


def _output_partition(task):
    path, arm, seed, n, width, sha = task
    side = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == set(side['arrays']) == {'img'}, (path, 'array keys')
        x = z['img']
    want = dict(list_sha256=sha, arm=arm, aspect='genre', seed=seed,
                rows=n, width=width, dtype='float16', scale='full', kind='held')
    for key, value in want.items():
        assert side[key] == value, (path, key, side[key], value)
    record = side['arrays']['img']
    assert record == dict(shape=list(x.shape), dtype=str(x.dtype), sha256=array_sha(x)), (path, 'array record')
    _unit_array(x, n, width, path)
    return r20_partition(x)


def derive(outputs, arm, inputs=None, draws=2000, workers=8, check_k4=False):
    """Read raw inputs and independently compute everything before any verdict read."""
    inp = inputs or Inputs()
    assert 1 <= workers <= 8
    assert arm and Path(arm).name == arm and arm not in ('.', '..')
    outputs = Path(outputs)
    held = json.loads(inp.held.read_text())
    assert set(held) == {'names', 'sha256'}
    names = held['names']
    assert len(names) == len(set(names)) == inp.n and names == sorted(names)
    assert sha256_names(names) == held['sha256'] == inp.held_sha, 'held hash/order'
    labels = genre_labels(names, inp.annotations, inp.genre_csvs)
    assert int((labels >= 0).sum()) == inp.labelled, 'labelled genre count'
    sources = {}
    def record(path):
        sources[str(path)] = file_sha(path)
    for path in (inp.held, inp.annotations, *inp.genre_csvs):
        record(path)

    meta = json.loads((inp.teacher / 'meta.json').read_text())
    assert meta['held_sha256'] == inp.held_sha
    assert meta['instruction'] == 'Describe the genre of this artwork in one word.'
    assert meta['model_id'] == 'Qwen/Qwen3-VL-Embedding-2B' and len(meta['model_revision']) == 40
    assert meta['shape'] == [inp.n, inp.width] and meta['dtype'] == 'float16'
    with np.load(inp.teacher / 'ids.npz', allow_pickle=False) as z:
        assert z.files == ['painting_names'] and z['painting_names'].tolist() == names
    x = np.load(inp.teacher / 'img.npy', allow_pickle=False)
    _unit_array(x, inp.n, inp.width, inp.teacher)
    assert array_sha(x) == meta['img_sha256'], 'teacher array hash'
    t = r20_partition(x)
    del x
    for name in ('meta.json', 'ids.npz', 'img.npy'):
        record(inp.teacher / name)

    gbase = inp.rb / 'full/held/K4' if check_k4 else outputs / 'stage_c/full/held' / arm
    tasks = [(base / f'genre__s{s}.npz', name, s, inp.n, inp.width, inp.held_sha)
             for base, name in ((gbase, arm), (inp.rb / 'full/held/K4', 'K4')) for s in SEEDS]
    # K4 self-check needs each physical output clustered only once.
    unique = list(dict.fromkeys(tasks))
    if workers == 1:
        values = list(map(_output_partition, unique))
    else:
        with mp.get_context('fork').Pool(workers) as pool:
            values = pool.map(_output_partition, unique, chunksize=1)
    parts = dict(zip(unique, values))
    g = {s: parts[tasks[s]] for s in SEEDS}
    r = {s: parts[tasks[5 + s]] for s in SEEDS}
    for path, *_ in unique:
        record(path)
        record(path.with_suffix('.json'))

    # Comparison uses recomputed arrays, never cached arrays as statistic input.
    checks = [(inp.rb / 'partitions/full/T/genre__s0__r20.npz', t)]
    for role, ps in [('G', g), ('R', r)]:
        folder = (outputs / 'stage_c/partitions' / arm if role == 'G' and not check_k4
                  else inp.rb / 'partitions/full/K4')
        checks.extend((folder / f'genre__s{s}__r20.npz', ps[s]) for s in SEEDS)
    differences = []
    checked = 0
    for path, mine in checks:
        if not path.is_file():
            differences.append(f'{path}: missing cached partition')
            sources[str(path)] = None
            continue
        with np.load(path, allow_pickle=False) as z:
            assert 'labels' in z.files, (path, 'partition key')
            theirs = z['labels']
        assert theirs.shape == (3, inp.n) and theirs.dtype.kind in 'iu', (path, 'partition shape/dtype')
        record(path)
        checked += 1
        if not np.array_equal(mine, theirs):
            differences.append(f'{path}: unequal partitions ({int(np.count_nonzero(mine != theirs))} positions)')
    result = calculate(labels, g, t, r, draws=draws, workers=workers)
    result.update(arm=arm, stage='C', held_sha256=inp.held_sha, rule_commit=rg_paths.RULE_COMMIT,
                  G_half_frozen=.0263670169701253, labels_sha256=array_sha(labels),
                  label_counts={str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                  partitions=dict(checked=checked, differences=differences), inputs_sha256=sources)
    return result


def compare_k4(mine, report):
    """Acceptance check against exactly the reported K4 genre margin T."""
    differences = list(mine['partitions']['differences'])
    deltas = []
    assert report['held_sha256'] == mine['held_sha256'], 'reported held hash'
    margin = report['blocks']['K4']['aspects']['genre']['margins']['T']
    expected = dict(A_G=margin['A_cand'], A_T=margin['A_e']['T-r20'],
                    d=margin['d'], interval=margin['interval'])
    _numeric_differences({k: mine['d_T'][k] for k in expected}, expected, 'K4.genre.T', differences, deltas)
    _numeric_differences(mine['bootstrap']['resamples'], margin['n_resamples'], 'resamples', differences, deltas)
    _numeric_differences(mine['G_half'], mine['G_half_frozen'], 'frozen half gap', differences, deltas)
    return dict(agree=not differences, differences=differences,
                max_abs_diff=float(max(deltas, default=0)), tolerance=TOL)


def _code_record():
    root = rg_paths.ROUND.parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return dict(code_commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                files={str(p): file_sha(p) for p in
                       (Path(__file__), rg_paths.ROUND / 'rg_paths.py',
                        rg_paths.TEST / '20261025_recipe_backbones/rb_v2.py')})


def run(outputs, arm=None, *, smoke=False, check_k4=False, workers=8, inputs=None):
    outputs = Path(outputs).resolve()
    assert 1 <= workers <= 8
    if smoke or check_k4:
        assert rg_paths.smoke_ok(outputs), 'smoke/check outputs must be under /tmp/claude-0/'
    assert not (smoke and check_k4), '--check-k4 requires all 2000 draws'
    for frozen in (rg_paths.EG_OUT, rg_paths.RB_OUT, rg_paths.CT_OUT):
        assert not outputs.is_relative_to(frozen.resolve()), 'cannot write under frozen artifacts'
    inp = inputs or Inputs()
    if check_k4:
        arm = 'K4'
    assert arm, 'give --arm (the recorded stage B pick)'
    assert Path(arm).name == arm and arm not in ('.', '..')
    target = outputs / 'stage_c' / ('rederive_vc2_k4.json' if check_k4 else 'rederive_vc2.json')
    comparison_path = inp.rb / 'reported_stage4.json' if check_k4 else outputs / 'stage_c/verdict.json'
    assert comparison_path.is_file(), f'missing comparison file: {comparison_path}'
    config = dict(arm=arm, smoke=smoke, check_k4=check_k4, draws=20 if smoke else 2000,
                  inputs=json.loads(json.dumps(asdict(inp), default=str)), code=_code_record(),
                  comparison_sha256=file_sha(comparison_path))
    if target.is_file():
        old = json.loads(target.read_text())
        if old['config'] == config and all(
                (file_sha(p) if Path(p).is_file() else None) == sha
                for p, sha in old['result']['inputs_sha256'].items()):
            return dict(agree=old['agree'], file=str(target), skipped=True,
                        differences=old['differences'])

    start = time.monotonic()
    mine = derive(outputs, arm, inp, draws=config['draws'], workers=workers, check_k4=check_k4)
    mine['smoke'] = smoke
    # This is deliberately the first read of the other implementation's numbers.
    reported = json.loads(comparison_path.read_text())
    comparison = compare_k4(mine, reported) if check_k4 else compare_verdict(mine, reported)
    mine['inputs_sha256'][str(comparison_path)] = config['comparison_sha256']
    document = dict(schema=1, stage='C', arm=arm, smoke=smoke, check_k4=check_k4,
                    config=config, code_commit=config['code']['code_commit'],
                    code_dirty=config['code']['dirty'], rule_commit=rg_paths.RULE_COMMIT,
                    versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
                    seconds=round(time.monotonic() - start, 2), result=mine, **comparison)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix('.pending.json')
    pending.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    pending.replace(target)
    return dict(agree=document['agree'], file=str(target), seconds=document['seconds'],
                differences=document['differences'], max_abs_diff=document['max_abs_diff'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outputs', type=Path, default=rg_paths.OUTPUTS)
    parser.add_argument('--arm', help='stage B pick; required except in --check-k4')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-k4', action='store_true')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args(argv)
    rg_paths.check_imports()
    summary = run(args.outputs, args.arm, smoke=args.smoke, check_k4=args.check_k4, workers=args.workers)
    print(json.dumps({k: v for k, v in summary.items() if k != 'differences'}, sort_keys=True))
    for difference in summary['differences'][:10]:
        print('DIFFERENCE', difference)
    if len(summary['differences']) > 10:
        print(f"... {len(summary['differences'])} differences recorded in {summary['file']}")
    return 0 if summary['agree'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
