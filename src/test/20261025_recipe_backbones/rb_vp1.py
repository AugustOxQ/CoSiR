"""VP1 (rule §4.1, C13): independent re-derivation of part 1 from the rule's text.

Written from DECISION_RULE.md §3.1 and §4.1 alone: P's names, the raw annotations, the WikiArt genre CSVs and the
saved P exports. It recomputes the labels, the partitions, each e_a(c), each interval and the pick, then compares
them with `part1/partitions/` and `part1/pick.json` and writes `part1/rederive_vp1.json` (+ `.md`).
"""
import argparse
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASPECTS = ('emotion', 'style', 'genre')
REFERENCE = 'R0-5'
CANDIDATES = ('R0-3', 'R6-5', 'R6-3', 'R7-5', 'R7-3')      # the rule's tie order
RECIPES = (REFERENCE,) + CANDIDATES
SEED_RANGES = {'emotion': 0.0046, 'style': 0.0082, 'genre': 0.0128}
TOL = 1e-6
N_P = 29232
WIDTH = 2048
N_DRAWS = 2000
SEEDS = (0, 1, 2)


def sha256_names(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def sha256_labels(labels):
    return hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).hexdigest()


# ---- labels (rule §3.1 "Labels") ------------------------------------------------------------------------------

def labels_from_annotations(names, annotations, genre_ids):
    """{aspect: int64[len(names)]}. Emotion: plurality of the painting's caption emotions, ties to the smallest
    code (codes = sorted names over the annotations, "something else" included). Style: the first caption's, codes =
    sorted style names. Genre: `genre_ids` (the WikiArt CSV id per name, -1 = none)."""
    emotion_names = sorted({r['emotion'] for r in annotations})
    style_names = sorted({r['art_style'] for r in annotations})
    emotion_code = {n: i for i, n in enumerate(emotion_names)}
    style_code = {n: i for i, n in enumerate(style_names)}
    wanted = set(names)
    counts, first_style = {}, {}
    for row in annotations:
        p = row['painting']
        if p not in wanted:
            continue
        if p not in counts:
            counts[p] = np.zeros(len(emotion_names), np.int64)
            first_style[p] = style_code[row['art_style']]
        counts[p][emotion_code[row['emotion']]] += 1
    assert wanted <= set(counts), f'{len(wanted - set(counts))} names without annotations'
    genre_ids = np.asarray(genre_ids)
    assert genre_ids.shape == (len(names),)
    return dict(
        emotion=np.array([int(np.argmax(counts[p])) for p in names], np.int64),   # argmax = first max = smallest code
        style=np.array([first_style[p] for p in names], np.int64),
        genre=genre_ids.astype(np.int64))


# ---- partitions and scores (rule §3.1) ------------------------------------------------------------------------

def partitions(x):
    """int64 [3, n]: MiniBatchKMeans k = 20, states 0, 1, 2, n_init 3, batch 4096, on the float32 cast of x."""
    x = np.asarray(x).astype(np.float32)
    norm = np.linalg.norm(x.astype(np.float64), axis=1)
    assert np.all(np.abs(norm - 1) < 1e-3), f'rows not unit norm (max error {np.abs(norm - 1).max():.4g})'
    out = []
    with threadpool_limits(limits=8):
        for state in SEEDS:
            km = MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096)
            out.append(km.fit(x).labels_.astype(np.int64))
    return np.stack(out)


def mean_ami(labels, parts, idx):
    """A_a(c, I): the mean over the partitions of AMI(labels[I], π[I]), default arguments."""
    return float(np.mean([adjusted_mutual_info_score(labels[idx], p[idx]) for p in parts]))


# ---- conditions, pick, triggers (rule §3.1, §4.1) -------------------------------------------------------------

def gain(e, lo, aspect):
    """Gain on an aspect ("at least small"): the interval's lower bound > 0 and e >= r_a."""
    return bool(lo > 0 and e >= SEED_RANGES[aspect])


def evaluate(effects):
    """Guard, pass, mean gain g and the flag for one candidate; `effects[aspect]` has e, lo, hi."""
    guard = all(effects[a]['e'] >= -SEED_RANGES[a] for a in ASPECTS)
    gains = {a: gain(effects[a]['e'], effects[a]['lo'], a) for a in ASPECTS}
    g = float(sum(effects[a]['e'] / SEED_RANGES[a] for a in ASPECTS) / 3)
    return dict(guard=bool(guard), gains=gains, **{'pass': bool(guard and any(gains.values()))}, g=g,
                flag_g_le_0=bool(g <= 0))


def decide(effects):
    """(candidates, outcome, recipe). `effects[c]` is None for a candidate excluded by VP2; it never passes."""
    cands, best = {}, None
    for c in CANDIDATES:
        if effects[c] is None:
            cands[c] = dict(guard=None, **{'pass': False, 'g': None, 'flag_g_le_0': None})
            continue
        ev = evaluate(effects[c])
        cands[c] = {k: ev[k] for k in ('guard', 'pass', 'g', 'flag_g_le_0')}
        if ev['pass'] and (best is None or round(ev['g'], 2) > round(cands[best]['g'], 2)):
            best = c                      # strict '>' keeps the first in the order on equal rounded values
    recipe = best or REFERENCE
    return cands, ('P-KEEP' if best is None else f'P-{best}'), recipe


def triggers(recipe):
    """K4 exists unless P-KEEP; K5 only under P-R7-5 or P-R7-3."""
    return {'K4': recipe != REFERENCE, 'K5': recipe in ('R7-5', 'R7-3')}


# ---- effects and intervals ------------------------------------------------------------------------------------

_STATE = {}


def _draw_chunk(task):
    """e_b for draws start..stop-1 of the aspect's own default_rng(0) stream (earlier draws are generated and dropped)."""
    start, stop = task
    labels, parts, n = _STATE['labels'], _STATE['parts'], len(_STATE['labels'])
    rng = np.random.default_rng(0)
    for _ in range(start):
        rng.integers(0, n, size=n)
    names = list(parts)
    out = np.zeros((len(names), stop - start))
    with threadpool_limits(limits=1):
        for b in range(stop - start):
            idx = rng.integers(0, n, size=n)
            a = {c: mean_ami(labels, parts[c], idx) for c in names}
            for j, c in enumerate(names):
                out[j, b] = a[c] - a[REFERENCE]
    return out


def score_effects(labels, parts, n_draws=N_DRAWS, workers=8):
    """{recipe: {A, e, lo, hi}} for one aspect. `labels` int64[n_L] and `parts[recipe]` int64[3, n_L] are already
    restricted to L_a; `parts` holds the reference and every non-excluded candidate."""
    n = len(labels)
    all_idx = np.arange(n)
    a_all = {c: mean_ami(labels, p, all_idx) for c, p in parts.items()}
    _STATE.update(labels=labels, parts=parts)
    edges = np.linspace(0, n_draws, min(workers, n_draws) + 1).astype(int)
    tasks = [(int(s), int(t)) for s, t in zip(edges[:-1], edges[1:]) if t > s]
    if workers == 1:
        chunks = [_draw_chunk(t) for t in tasks]
    else:
        with multiprocessing.get_context('fork').Pool(len(tasks)) as pool:
            chunks = pool.map(_draw_chunk, tasks)
    e_b = np.concatenate(chunks, axis=1)
    out = {}
    for j, c in enumerate(parts):
        lo, hi = np.percentile(e_b[j], [2.5, 97.5])
        out[c] = dict(A=a_all[c], e=a_all[c] - a_all[REFERENCE], lo=float(lo), hi=float(hi))
    return out


# ---- re-derivation from the saved files -----------------------------------------------------------------------

def _load_p(outputs, n_rows):
    p = json.loads((Path(outputs) / 'part1' / 'P.json').read_text())
    names = p['names']
    assert len(names) == n_rows and names == sorted(names) and len(set(names)) == n_rows, 'P names'
    assert sha256_names(names) == p['sha256'], 'P.json sha256 does not match its names'
    return names, p['sha256']


def _default_annotations():
    from src.data.artelingo import ANNOTATIONS_PATH
    with Path(ANNOTATIONS_PATH).open() as f:
        return json.load(f)


def _default_genre_loader(names):
    from src.data.wikiart_genre import load_wikiart_genre
    return load_wikiart_genre(np.asarray(names))


def _export_path(outputs, recipe, aspect):
    return Path(outputs) / 'part1' / 'train' / recipe / f'{aspect}__s0__P.npz'


def _read_export(outputs, recipe, aspect, p_sha):
    """A P export's image rows, after its training record names the same P (rows in P.json order, C6)."""
    path = _export_path(outputs, recipe, aspect)
    record = json.loads((path.parent / f'{aspect}__s0.json').read_text())
    assert record['p_sha256'] == p_sha, f'{path}: its record names another P ({record["p_sha256"]})'
    return np.load(path)['img']


def _check_export(x, n_rows, width=WIDTH):
    """The part of VP2 this module can check on its own: dtype, shape, finite, unit norm. Returns a problem or None."""
    if x.dtype != np.float16:
        return f'dtype {x.dtype}'
    if x.shape != (n_rows, width):
        return f'shape {x.shape}'
    x32 = x.astype(np.float32)
    if not np.isfinite(x32).all():
        return 'nonfinite rows'
    err = float(np.abs(np.linalg.norm(x32.astype(np.float64), axis=1) - 1).max())
    return None if err < 1e-3 else f'norm error {err:.4g}'


def _excluded_check(outputs, recipe, p_sha, n_rows, width):
    """A candidate the pick file excludes: are all three P exports there, and do they pass this module's checks?"""
    if not all(_export_path(outputs, recipe, a).exists() for a in ASPECTS):
        return dict(present=False, problems=[])
    problems = []
    for a in ASPECTS:
        bad = _check_export(_read_export(outputs, recipe, a, p_sha), n_rows, width)
        if bad:
            problems.append(f'{a}: {bad}')
    return dict(present=True, problems=problems)


def derive(outputs, annotations=None, genre_loader=None, excluded=(), n_draws=N_DRAWS, workers=8, n_rows=N_P,
           width=WIDTH):
    """Everything VP1 recomputes. `excluded` = candidates the pick file excludes by VP2: not scored, but their
    exports, when all three exist, are checked (an exclusion of exports that pass is a disagreement)."""
    assert REFERENCE not in excluded, 'R0-5 excluded: part 1 stops (rule §4.1 VP3)'
    names, p_sha = _load_p(outputs, n_rows)
    annotations = _default_annotations() if annotations is None else annotations
    genre_loader = _default_genre_loader if genre_loader is None else genre_loader
    labels = labels_from_annotations(names, annotations, genre_loader(names))
    label_sha = {a: sha256_labels(labels[a]) for a in ASPECTS}
    label_counts = {a: int((labels[a] >= 0).sum()) for a in ASPECTS}
    parts, problems, excluded_exports = {}, {}, {}
    for r in RECIPES:
        if r in excluded:
            excluded_exports[r] = _excluded_check(outputs, r, p_sha, n_rows, width)
            continue
        parts[r] = {}
        for a in ASPECTS:
            x = _read_export(outputs, r, a, p_sha)
            bad = _check_export(x, n_rows, width)
            if bad:
                problems.setdefault(r, []).append(f'{a}: {bad}')
                continue
            parts[r][a] = partitions(x)
    assert REFERENCE not in problems, f'R0-5 fails VP2: {problems[REFERENCE]}'
    usable = [r for r in RECIPES if r in parts and r not in problems]
    effects = {r: None for r in RECIPES}
    for r in usable:
        effects[r] = {}
    for a in ASPECTS:
        keep = labels[a] >= 0
        out = score_effects(labels[a][keep], {r: parts[r][a][:, keep] for r in usable}, n_draws, workers)
        for r in usable:
            effects[r][a] = out[r]
    for r in usable:
        for a in ASPECTS:
            effects[r][a]['gain'] = gain(effects[r][a]['e'], effects[r][a]['lo'], a)
    cands, outcome, recipe = decide({r: effects[r] for r in CANDIDATES})
    return dict(p_sha256=p_sha, label_sha256=label_sha, label_counts=label_counts, parts=parts, effects=effects,
                candidates=cands, outcome=outcome, recipe=recipe, triggers=triggers(recipe),
                vp2_problems=problems, excluded=sorted(excluded), excluded_exports=excluded_exports, n_draws=n_draws,
                versions={'sklearn': sklearn.__version__, 'numpy': np.__version__})


def _num(mine, pick, tol=TOL):
    if mine is None or pick is None:
        return dict(mine=mine, pick=pick, diff=None, agree=mine is pick)
    d = abs(float(mine) - float(pick))
    return dict(mine=float(mine), pick=float(pick), diff=d, agree=bool(d <= tol))


def _eq(mine, pick):
    return dict(mine=mine, pick=pick, agree=bool(mine == pick))


def pick_exclusions(pick):
    """Candidates the pick file lists as excluded by VP2."""
    out = set(pick.get('vp2_failed', {}))
    for c, block in pick['candidates'].items():
        if block.get('vp2_ok') is False or block.get('vp2_problems'):
            out.add(c)
    out |= {c for c in CANDIDATES if pick['recipes'].get(c) is None}
    return sorted(out)


def compare(outputs, mine):
    """Compare `mine` (from derive) with part1/pick.json and part1/partitions/. A candidate excluded by the pick file
    is excluded here too; everything else is compared."""
    outputs = Path(outputs)
    pick = json.loads((outputs / 'part1' / 'pick.json').read_text())
    excl = pick_exclusions(pick)
    res = dict(excluded=excl, excluded_exports=mine['excluded_exports'],
               draws=_eq(mine['n_draws'], pick['n_draws']), p_sha256=_eq(mine['p_sha256'], pick['p_sha256']),
               seed_ranges=_eq(SEED_RANGES, pick['seed_ranges']),
               label_counts={a: _eq(mine['label_counts'][a], pick['label_counts'][a]) for a in ASPECTS},
               label_sha256={a: _eq(mine['label_sha256'][a], pick['label_sha256'][a]) for a in ASPECTS},
               partitions={}, effects={}, candidates={})
    bad = []

    def note(path, entry):
        if not entry['agree']:
            bad.append(path)

    for k in ('draws', 'p_sha256', 'seed_ranges'):
        note(k, res[k])
    for c in excl:
        chk = mine['excluded_exports'].get(c)
        if chk is not None and chk['present'] and not chk['problems']:
            bad.append(f'pick excluded {c}, its exports pass VP2')
    for a in ASPECTS:
        note(f'label_counts/{a}', res['label_counts'][a])
        note(f'label_sha256/{a}', res['label_sha256'][a])
    for r in RECIPES:
        if r in excl:
            continue
        res['partitions'][r], res['effects'][r] = {}, {}
        if mine['effects'][r] is None:
            bad.append(f'{r}: mine failed VP2 {mine["vp2_problems"].get(r)} but the pick file scored it')
            continue
        for a in ASPECTS:
            f = outputs / 'part1' / 'partitions' / r / f'{a}.npz'
            if not f.exists():
                res['partitions'][r][a] = dict(equal=False, n_diff=None, missing=True)
            else:
                saved = np.load(f)['labels']
                ok = saved.shape == mine['parts'][r][a].shape
                res['partitions'][r][a] = dict(equal=bool(ok and np.array_equal(saved, mine['parts'][r][a])),
                                               n_diff=int((saved != mine['parts'][r][a]).sum()) if ok else None)
            if not res['partitions'][r][a]['equal']:
                bad.append(f'partitions/{r}/{a}')
            m, p = mine['effects'][r][a], pick['recipes'][r][a]
            res['effects'][r][a] = dict(A=_num(m['A'], p['A']), e=_num(m['e'], p['e']), lo=_num(m['lo'], p['lo']),
                                        hi=_num(m['hi'], p['hi']), gain=_eq(m['gain'], p['gain']))
            for k in ('e', 'lo', 'hi', 'gain'):          # A is shown beside them; e = A - A(R0-5) carries it
                note(f'effects/{r}/{a}/{k}', res['effects'][r][a][k])
    for c in CANDIDATES:
        if c in excl:
            continue
        m, p = mine['candidates'][c], pick['candidates'][c]
        res['candidates'][c] = dict(guard=_eq(m['guard'], p['guard']), **{'pass': _eq(m['pass'], p['pass'])},
                                    g=_num(m['g'], p['g']), flag_g_le_0=_eq(m['flag_g_le_0'], p['flag_g_le_0']))
        for k, v in res['candidates'][c].items():
            note(f'candidates/{c}/{k}', v)
    res['outcome'] = _eq(mine['outcome'], pick['outcome'])
    res['recipe'] = _eq(mine['recipe'], pick['recipe'])
    res['triggers'] = _eq(mine['triggers'], pick['triggers'])
    for k in ('outcome', 'recipe', 'triggers'):
        note(k, res[k])
    res['disagreements'] = bad
    res['agree'] = not bad
    return res


def _md(res):
    lines = ['# VP1: part 1 re-derivation (independent)', '',
             f'Agree: **{res["agree"]}** (tolerance {TOL:g}; {res["n_draws"]} bootstrap draws). '
             f'Pick: mine {res["outcome"]["mine"]}, pick file {res["outcome"]["pick"]}.', '']
    lines.append(f'Excluded by VP2 (from the pick file): {res["excluded"] or "none"}.')
    lines += ['', 'Disagreements: ' + ('none.' if not res['disagreements'] else '')]
    lines += [f'- {d}' for d in res['disagreements']]
    lines += ['', '| recipe | aspect | e (mine) | e (pick) | lo diff | hi diff |', '|---|---|---|---|---|---|']
    for r, per in res['effects'].items():
        for a, v in per.items():
            lines.append(f'| {r} | {a} | {v["e"]["mine"]:.6f} | {v["e"]["pick"]:.6f} | {v["lo"]["diff"]:.2g} | '
                         f'{v["hi"]["diff"]:.2g} |')
    return '\n'.join(lines) + '\n'


def run(outputs, annotations=None, genre_loader=None, n_draws=N_DRAWS, workers=8, n_rows=N_P, width=WIDTH):
    """Re-derive, compare, write part1/rederive_vp1.json and .md. Returns the result dict."""
    outputs = Path(outputs)
    pick = json.loads((outputs / 'part1' / 'pick.json').read_text())
    mine = derive(outputs, annotations, genre_loader, excluded=pick_exclusions(pick), n_draws=n_draws,
                  workers=workers, n_rows=n_rows, width=width)
    res = compare(outputs, mine)
    res.update(n_draws=n_draws, versions=dict(mine=mine['versions'], pick=pick.get('versions')),
               vp2_problems_mine=mine['vp2_problems'], mine=dict(
                   outcome=mine['outcome'], recipe=mine['recipe'], triggers=mine['triggers'],
                   candidates=mine['candidates'],
                   effects={r: v for r, v in mine['effects'].items()}))
    (outputs / 'part1' / 'rederive_vp1.json').write_text(json.dumps(res, indent=1))
    (outputs / 'part1' / 'rederive_vp1.md').write_text(_md(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outputs', default=None, help='default: rb_paths.OUTPUTS')
    ap.add_argument('--smoke', action='store_true', help='20 draws; outputs must be under /tmp/claude-0/')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    if args.outputs is None:
        import rb_paths
        args.outputs = rb_paths.OUTPUTS
    if args.smoke and not str(Path(args.outputs).resolve()).startswith('/tmp/claude-0/'):
        sys.exit('--smoke refuses an outputs path outside /tmp/claude-0/')
    res = run(args.outputs, n_draws=20 if args.smoke else N_DRAWS, workers=args.workers)
    print(f'VP1 agree={res["agree"]}; outcome mine={res["outcome"]["mine"]} pick={res["outcome"]["pick"]}; '
          f'{len(res["disagreements"])} disagreements')
    return 0 if res['agree'] else 1


if __name__ == '__main__':
    sys.exit(main())
