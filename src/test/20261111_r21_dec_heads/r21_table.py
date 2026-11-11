"""r21 stage 2 change table and diagnostics (ticket 06; spec §3.4, §3.6, D5, D13, D14, SC2 to SC4).

CLI (from the worktree root; Python of the CoSiR env; `--out` defaults to the real `scores/` folder, tests and smoke
runs pass their own):
  python src/test/20261111_r21_dec_heads/r21_table.py table [--arms A ...] [--root R] [--cstar-root R10_FULL]
      For each arm (default: dec-off and the three stage 1b picks of `scores/stage1_picks.json`) against saved c*
      (r10's `full/held/k11=40/`), paired by seed 0 to 4, per aspect:
        AMI change at the class count (9, 27, 10) and at 20 clusters, each with `rs_ablation.held_effect` (report
        2026-12-28: painting and seed bootstrap, 2,000 draws, `held_draws`' default_rng(0) per aspect);
        mAP change with `r20_ret.paired` (report 2027-01-13: shared query and seed bootstrap, 10,000 draws);
        the word per cell (D14): "raises" when the interval is above zero (lo > 0), "lowers" when below (hi < 0), else
        "no clear change"; an interval that touches zero is "no clear change". (The reference code has no such word;
        its nearest rule, `ns_ret._word`, is strict in the same way: WIN needs lo > 0.)
        The absolute values of the arm and of c* beside, per seed too. No correction across cells.
      -> <out>/stage2_table.json and .md (`--name` changes the stem). Held (test) exports are read by design (stage 2).
  python .../r21_table.py diag [--arms A ...] [--root R]
      Spec §3.6, decides nothing: cosine silhouette of each arm's test embeddings against its own k-means partition at
      the class count (state 0), per aspect and seed, also for c*, the plain Qwen image and the teacher; largest and
      smallest cluster share; DEC epochs, stop epoch and reason, the fraction away from the DEC-start clusters (training
      sidecars); validation AMI and mAP of the arms on seeds 0 to 4 (ticket 05's `val_cached`). -> <out>/diag.json, .md
  python .../r21_table.py regress --table FILE
      Compares a table made with a saved removal arm (r10's tree) with r10's `ablation/held.json` (AMI at 20) and r20's
      `ablation_ret.json` (mAP), per cell.

One code path with gate 0a: the class-count AMI uses `ns_metrics.partitions` (r17's clustering), the same function that
gives r10's partitions at 20 (checked equal to r10's saved partitions: the MiniBatchKMeans settings, unit float32 rows,
states 0 to 2 and 8 threads are the same in `ct_score.output_partitions` and `ns_metrics._fit_states`). The mAP uses
`r20_ret.unit`, `score_rows` and `paired`. Exports are loaded by `rg_held.load_held_output` (r10's VC3 checks).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(HERE))
import r21_arms  # noqa: E402
import r21_score as SC  # noqa: E402

ASPECTS = SC.ASPECTS
SEEDS = SC.SEEDS
SET = SC.SET
N_HELD = 12281
FIXED_K = 20                                      # report 2026-12-28's count
N_AMI_DRAWS = 2000                                # report 2026-12-28
N_MAP_DRAWS = 10000                               # report 2027-01-13
CSTAR_ARM = 'k11=40'                              # saved c*: r10's folder name
R21_FULL = SC.R21 / 'full'
R10_FULL = SC.R10_FULL
SCORES = SC.SCORES
TOL_REG = 1e-9                                    # the regression: ours and the saved figure, per cell
FORMS = ('plain', 'start', 'teacher')
R10_HELD_RECORD = SC.R10_FULL.parent / 'ablation' / 'held.json'


def _deps():
    global rs_ablation, rg_held, rb_score, rs_paths, rb_part2_report, r20_ret, ns_metrics, ns_data, ns_paths
    if 'rs_ablation' in globals():
        return
    SC._deps()
    import rs_ablation, rg_held, rb_score, rs_paths, rb_part2_report  # noqa: E401,F811
    r20_ret, ns_metrics, ns_data, ns_paths = SC.r20_ret, SC.ns_metrics, SC.ns_data, SC.ns_paths


def held_sha():
    _deps()
    return rs_paths.HELD_SHA256


# ================================================================================================ the word rule (D14)
def word(lo, hi):
    """D14: "raises" when the whole interval is above zero, "lowers" when below, else "no clear change". Strict: an
    interval with an end at zero is "no clear change"."""
    assert lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo <= hi, (lo, hi)
    if lo > 0:
        return 'raises'
    if hi < 0:
        return 'lowers'
    return 'no clear change'


def cell(change, lo, hi):
    """One cell: the change, its 95% interval and the word."""
    assert np.isfinite(change), change
    return dict(change=float(change), lo=float(lo), hi=float(hi), word=word(lo, hi))


# ================================================================================================ the exports
def held_path(root, arm, aspect, seed):
    return Path(root) / 'held' / arm / f'{aspect}__s{seed}.npz'


def _sidecar_meta(path):
    """The sidecar's fields that pair an export with c*'s (the arrays are checked against them by `load_held`)."""
    path = Path(path)
    assert path.is_file() and path.with_suffix('.json').is_file(), f'{path}: export or sidecar missing'
    side = json.loads(path.with_suffix('.json').read_text())
    return dict(arm=side['arm'], aspect=side['aspect'], seed=side['seed'], rows=side['rows'],
                list_sha256=side['list_sha256'], array_sha256=side['arrays']['img']['sha256'], path=str(path))


def read_metas(root, arm):
    """{(aspect, seed): sidecar meta} of an arm's 15 held exports under `root/held/<arm>`; a missing file is refused."""
    return {(a, s): _sidecar_meta(held_path(root, arm, a, s)) for a in ASPECTS for s in SEEDS}


def check_pairing(arm_metas, cstar_metas, expected_sha):
    """The arm and c* must hold the same grid (every aspect, seeds 0 to 4, nothing else), each entry labelled as its key,
    12,281 rows in the held list's order (the sidecar binds the list's SHA-256, `expected_sha`), and not be the same
    files under two names. A missing seed, a mismatched held order or a mislabelled sidecar is refused."""
    grid = {(a, s) for a in ASPECTS for s in SEEDS}
    names = {}
    for what, metas in (('arm', arm_metas), ('c*', cstar_metas)):
        assert set(metas) == grid, f'{what}: grid differs; missing {sorted(grid - set(metas))}, extra {sorted(set(metas) - grid)}'
        for (a, s), m in metas.items():
            assert (m['aspect'], m['seed']) == (a, s), f'{what} {a} s{s}: the sidecar says {m["aspect"]} s{m["seed"]}'
            assert m['rows'] == N_HELD, f'{what} {a} s{s}: {m["rows"]} rows'
            assert m['list_sha256'] == expected_sha, f'{what} {a} s{s}: held order differs from the held list'
        arms = {m['arm'] for m in metas.values()}
        assert len(arms) == 1, f'{what}: more than one arm name in the sidecars: {arms}'
        names[what] = arms.pop()
    if names['arm'] != names['c*']:
        same = [k for k in grid if arm_metas[k]['array_sha256'] == cstar_metas[k]['array_sha256']]
        assert not same, f'{names["arm"]} and {names["c*"]} hold the same array at {sorted(same)[:3]}'


def load_held(root, arm, aspect, seed):
    """(float32 [12,281, 2048] rows, sidecar meta) of one export, through r10's checked loader (`rg_held.load_held_output`:
    key `img`, float16, finite, unit rows, the held list's SHA-256, arm, seed, aspect and array hash in the sidecar)."""
    _deps()
    path = held_path(root, arm, aspect, seed)
    meta = _sidecar_meta(path)
    x = rg_held.load_held_output(path, arm, seed, held_sha(), aspect=aspect)
    assert x.shape == (N_HELD, 2048) and (meta['arm'], meta['aspect'], meta['seed']) == (arm, aspect, seed)
    return x, meta


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# ================================================================================================ partitions (cached)
def get_partitions(cache_dir, arm, aspect, seed, k, meta, x):
    """int [3, n] MiniBatchKMeans partitions (states 0 to 2) of one export at `k` clusters: `ns_metrics.partitions` (r17's
    procedure; r10's at 20, checked equal to r10's saved ones). Cached under `<cache_dir>/partitions/<arm>/` with the
    input's array hash in the identity; with a cached file `x` may be None. `seed` None is a seedless source."""
    _deps()
    ident = dict(array_sha256=meta['array_sha256'], k=int(k), rows=int(meta['rows']), states=list(ns_paths.KMEANS_STATES),
                 n_init=ns_metrics.N_INIT, batch=ns_metrics.BATCH, code='ns_metrics.partitions')
    path = Path(cache_dir) / 'partitions' / arm / f'{aspect}__s{"-" if seed is None else seed}__k{k}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert json.loads(str(z['identity'])) == ident, f'{path}: the cached partitions were made from other inputs'
            labels = z['labels']
    else:
        assert x is not None, f'{path}: not cached and no array given'
        labels = ns_metrics.partitions(x, k, SET)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.pending.npz')
        np.savez_compressed(tmp, labels=labels, identity=np.array(json.dumps(ident)))
        tmp.replace(path)
    assert labels.shape == (3, meta['rows']) and np.issubdtype(labels.dtype, np.integer), (path, labels.shape)
    return labels


# ================================================================================================ the cells
def ami_cell(arm_parts, cstar_parts, y, n_draws, workers):
    """AMI change, arm minus c*, with report 2026-12-28's bootstrap: `rs_ablation.held_effect` on fixed partitions
    ({seed: int [3, n]} over seeds 0 to 4; the same painting and seed draws for both; `default_rng(0)`). `y` int [n], < 0
    masked inside the score. The partitions can be at any k (the function's key 'r20' is only a name)."""
    _deps()
    fixed = lambda parts: {s: {'r20': p} for s, p in parts.items()}
    e = rs_ablation.held_effect(fixed(arm_parts), fixed(cstar_parts), np.asarray(y), n_draws, workers)
    return dict(cell(e['effect'], e['lo'], e['hi']), arm=e['arm_ami'], cstar=e['final_ami'],
                arm_seed=e['arm_seed_ami'], cstar_seed=e['final_seed_ami'], n_draws=int(n_draws))


def map_cell(arm, cstar, n_draws):
    """mAP change, arm minus c*, with report 2027-01-13's bootstrap: `r20_ret.paired` (shared query and seed draws,
    `default_rng(0)`). `arm` and `cstar` are `score_aps` dicts; the queries must be the same ones in the same order."""
    _deps()
    assert np.array_equal(arm['query_ids'], cstar['query_ids']), 'the query paintings differ between arm and c*'
    assert arm['dropped'] == cstar['dropped'], 'the dropped query counts differ (dropped)'
    p = r20_ret.paired(arm, cstar, n_draws)
    per = lambda r: r['ap'].mean(axis=1)
    return dict(cell(p['d'], p['lo'], p['hi']), arm=float(np.mean(per(arm))), cstar=float(np.mean(per(cstar))),
                arm_seed={str(s): float(v) for s, v in zip(SEEDS, per(arm))},
                cstar_seed={str(s): float(v) for s, v in zip(SEEDS, per(cstar))},
                n_queries=int(arm['ap'].shape[1]), dropped=int(arm['dropped']), n_draws=int(n_draws))


# ================================================================================================ arms
def check_arm_names(arms, root):
    """Arms under the real r21 tree must be `dec-off` or a name `r21_arms.parse_arm` accepts; any other tree (the regression
    reads r10's) takes its own folder names. No arm twice, and c* is not an arm."""
    assert arms and len(set(arms)) == len(arms), f'arms: {arms}'
    for a in arms:
        if Path(root) == R21_FULL:
            r21_arms.parse_arm(a)
        else:
            assert a, a


def picked_arms(picks_path):
    """`dec-off` and the stage 1b pick of each form (plain, start, teacher), from 05's picks file."""
    picks = json.loads(Path(picks_path).read_text())
    assert picks.get('stage1b'), f'{picks_path}: no stage 1b picks yet'
    assert set(picks['stage1b']) == set(FORMS), sorted(picks['stage1b'])
    return ['dec-off'] + [picks['stage1b'][f]['pick'] for f in FORMS]


# ================================================================================================ the table
def cstar_reference(aspect, k, ami_k, ami_20, map_val):
    """c*'s absolute values against the saved ones: AMI at the class count against r17's cells file (what gate 0a
    compares), AMI at 20 against r10's held record, mAP against r20's `ablation_ret.json`. Crashes when c* here is not
    the c* of those files: the changes would be against another comparator."""
    cells = {c['id']: c for c in json.loads(SC.R17_CLU.read_text())}
    r17 = cells[f'SC3/{SET}/qwen/{aspect}/ami/{k}/teacher']['cand']
    r10 = json.loads(R10_HELD_RECORD.read_text())
    finals = {r['aspects'][aspect]['final_ami'] for r in r10['results'].values() if r['status'] == 'ok'}
    assert len(finals) == 1, finals
    r20 = json.loads(SC.ABLATION_RET.read_text())['aspects'][aspect]['arms'][CSTAR_ARM]['map']
    out = dict(ami_class_count=dict(ours=ami_k, saved=r17, diff=ami_k - r17, tol=SC.TOL_AMI, source=str(SC.R17_CLU)),
               ami_20=dict(ours=ami_20, saved=finals.pop(), tol=TOL_REG, source=str(R10_HELD_RECORD)),
               map=dict(ours=map_val, saved=r20, tol=SC.TOL_MAP, source=str(SC.ABLATION_RET)))
    out['ami_20']['diff'] = ami_20 - out['ami_20']['saved']
    out['map']['diff'] = map_val - r20
    for key, c in out.items():
        c['ok'] = bool(abs(c['diff']) < c['tol'])
    assert all(c['ok'] for c in out.values()), f'{aspect}: c* differs from the saved figures: {out}'
    return out


def _mean_ami(y, parts_by_seed):
    return float(np.mean([ns_metrics.score_partitions('ami', y, p) for p in parts_by_seed.values()]))


def _score_set(root, arm, aspect, ks, cache, y_map, names, workers):
    """Everything one arm contributes for one aspect: partitions at each k per seed (cached), the AP matrix."""
    parts = {k: {} for k in ks}
    rows = []
    for s in SEEDS:
        x, meta = load_held(root, arm, aspect, s)
        for k in ks:
            parts[k][s] = get_partitions(cache, arm, aspect, s, k, meta, x)
        rows.append(SC.r20_ret.score_rows(SC.r20_ret.unit(x), y_map, names, workers))
    for r in rows[1:]:
        SC.r20_ret.same_queries(rows[0], r)
    ap = dict(ap=np.stack([r['ap'] for r in rows]), query_ids=rows[0]['query_ids'], dropped=rows[0]['dropped'])
    return parts, ap


def code_state():
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '*.py'], cwd=HERE, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except Exception:
        sha, dirty = None, None
    return dict(code_commit=sha, code_dirty=dirty, r21_table_sha256=_file_sha(__file__))


def table(arms, root, cstar_root, out, n_ami_draws=N_AMI_DRAWS, n_map_draws=N_MAP_DRAWS, workers=8, aspects=ASPECTS,
          name='stage2_table'):
    """The stage 2 change table; returns the record (written to `<out>/<name>.json` and `.md`)."""
    _deps()
    t0 = time.monotonic()
    out = Path(out)
    root, cstar_root = Path(root), Path(cstar_root)
    check_arm_names(arms, root)
    names, sha, labels = rb_part2_report.held_labels()
    names = np.asarray(names)
    assert len(names) == N_HELD and sha == held_sha()
    cache = out / 'cache'
    c_metas = read_metas(cstar_root, CSTAR_ARM)
    metas = {a: read_metas(root, a) for a in arms}
    for a in arms:
        check_pairing(metas[a], c_metas, sha)
    doc = dict(stage='r21_stage2_table', written=time.strftime('%Y-%m-%d %H:%M'), root=str(root), cstar_root=str(cstar_root),
               arms=list(arms), seeds=list(SEEDS), n_ami_draws=int(n_ami_draws), n_map_draws=int(n_map_draws),
               complete=bool(tuple(aspects) == ASPECTS and n_ami_draws == N_AMI_DRAWS and n_map_draws == N_MAP_DRAWS),
               k_fixed=FIXED_K, held_sha256=sha, comparator=CSTAR_ARM, pairing='arm seed s against c* seed s',
               draws=dict(ami='per aspect: default_rng(0); I_b over the 12,281 rows, then one S_b over seeds 0 to 4 shared '
                              'by arm and c* (rs_ablation.held_draws)',
                          map='default_rng(0), shared query and seed draws (ns_ret.paired_ret, comp_seeds=shared)'),
               correction='none across cells (as the existing ablation table)',
               inputs={a: {f'{asp}__s{s}': dict(file_sha256=_file_sha(m['path']), array_sha256=m['array_sha256'])
                           for (asp, s), m in metas[a].items()} for a in arms},
               cstar_inputs={f'{asp}__s{s}': dict(file_sha256=_file_sha(m['path']), array_sha256=m['array_sha256'])
                             for (asp, s), m in c_metas.items()},
               aspects={}, cstar_check={}, **code_state())
    for aspect in aspects:
        k = ns_paths.CLASS_COUNT[(SET, aspect)]
        y_ami = ns_data.labels(SET, 'held', aspect)
        y_map = np.asarray(labels[aspect], dtype=np.int64)
        assert np.array_equal(y_ami, y_map), f'{aspect}: ns_data and rb_part2_report labels differ'
        assert len(y_ami) == N_HELD and ns_metrics.class_count(y_ami) == k
        ks = (k, FIXED_K)
        c_parts, c_ap = _score_set(cstar_root, CSTAR_ARM, aspect, ks, cache, y_map, names, workers)
        doc['cstar_check'][aspect] = cstar_reference(aspect, k, _mean_ami(y_ami, c_parts[k]), _mean_ami(y_ami, c_parts[FIXED_K]),
                                                     float(c_ap['ap'].mean(axis=1).mean()))
        print(f'{aspect}: c* checked against the saved figures', flush=True)
        doc['aspects'][aspect] = dict(k=int(k), n_labelled=int((y_ami >= 0).sum()), arms={})
        for arm in arms:
            parts, ap = _score_set(root, arm, aspect, ks, cache, y_map, names, workers)
            res = dict(ami_class_count=ami_cell(parts[k], c_parts[k], y_ami, n_ami_draws, workers),
                       ami_20=ami_cell(parts[FIXED_K], c_parts[FIXED_K], y_ami, n_ami_draws, workers),
                       map=map_cell(ap, c_ap, n_map_draws))
            doc['aspects'][aspect]['arms'][arm] = res
            print(f'{aspect} {arm}: ' + '; '.join(f"{key} {c['change']:+.6f} [{c['lo']:+.6f}, {c['hi']:+.6f}] {c['word']}"
                                                   for key, c in res.items()), flush=True)
    doc['seconds'] = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    (out / f'{name}.md').write_text(table_markdown(doc))
    print(f'{doc["seconds"]:.0f}s -> {out / name}.json, .md', flush=True)
    return doc


def table_markdown(doc):
    f = lambda c: f"{c['change']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] {c['word']}"
    lines = ['# r21 stage 2 change table (ArtELingo test set, Qwen3-VL-Embedding-2B, seeds 0 to 4)', '',
             'Change = arm minus saved c* (r10 `k11=40`), paired by seed. AMI: unit rows, MiniBatchKMeans (3 inits, batch 4,096, '
             f'states 0 to 2), at the class count (emotion 9, style 27, genre 10) and at {doc["k_fixed"]} clusters; 95% interval from '
             f'the painting and seed bootstrap, {doc["n_ami_draws"]:,} draws (report 2026-12-28). mAP: image-to-image label mAP, plain '
             f'cosine, query excluded; shared query and seed bootstrap, {doc["n_map_draws"]:,} draws (report 2027-01-13). '
             'Word: "raises" when the interval is above zero, "lowers" when below, "no clear change" when it includes zero or '
             'touches it. No correction across cells.', '']
    if not doc['complete']:
        lines += ['**Not the full table** (a subset of aspects or a reduced draw count): not for the report.', '']
    for arm in doc['arms']:
        lines += [f'## {arm}', '', '| aspect | AMI at class count: change [95%] word | arm / c* | AMI at 20: change [95%] word | arm / c* '
                  '| mAP: change [95%] word | arm / c* |', '|---|---|---|---|---|---|---|']
        for a, r in doc['aspects'].items():
            c = r['arms'][arm]
            ab = lambda x: f"{x['arm']:.4f} / {x['cstar']:.4f}"
            lines.append(f"| {a} (k={r['k']}) | {f(c['ami_class_count'])} | {ab(c['ami_class_count'])} | {f(c['ami_20'])} | "
                         f"{ab(c['ami_20'])} | {f(c['map'])} | {ab(c['map'])} |")
        lines.append('')
    lines += ['c* here matches the saved figures: ' + '; '.join(
        f"{a}: AMI {v['ami_class_count']['diff']:+.1e} (r17), AMI20 {v['ami_20']['diff']:+.1e} (r10), mAP {v['map']['diff']:+.1e} (r20)"
        for a, v in doc['cstar_check'].items()), '']
    return '\n'.join(lines)


# ================================================================================================ regression check
def regress(table_path):
    """Per cell, our removal-arm change and interval against r10's held record (AMI at 20) and `ablation_ret.json` (mAP)."""
    doc = json.loads(Path(table_path).read_text())
    held = json.loads(R10_HELD_RECORD.read_text())
    ret = json.loads(SC.ABLATION_RET.read_text())
    by_arm = {r['arm']: r for r in held['results'].values() if r['status'] == 'ok'}
    rows, worst = [], 0.0
    for arm in doc['arms']:
        for aspect, r in doc['aspects'].items():
            ours = r['arms'][arm]
            if arm == CSTAR_ARM:                                  # c* against itself: zero everywhere, no clear change
                for key in ('ami_class_count', 'ami_20', 'map'):
                    c = ours[key]
                    ok = (c['change'], c['lo'], c['hi']) == (0.0, 0.0, 0.0) and c['word'] == 'no clear change'
                    rows.append((arm, aspect, key, 'self', (c['change'], c['lo'], c['hi']), (0.0, 0.0, 0.0), ok))
                    worst = max(worst, 0.0 if ok else 1.0)
                continue
            saved = by_arm[arm]['aspects'][aspect]
            pairs = [('ami_20', 'change', ours['ami_20']['change'], saved['effect']), ('ami_20', 'lo', ours['ami_20']['lo'], saved['lo']),
                     ('ami_20', 'hi', ours['ami_20']['hi'], saved['hi']), ('ami_20', 'arm', ours['ami_20']['arm'], saved['arm_ami']),
                     ('ami_20', 'cstar', ours['ami_20']['cstar'], saved['final_ami'])]
            comp = ret['aspects'][aspect]['arms'][arm]
            pairs += [('map', 'change', ours['map']['change'], comp['comparison']['d']), ('map', 'lo', ours['map']['lo'], comp['comparison']['lo']),
                      ('map', 'hi', ours['map']['hi'], comp['comparison']['hi']), ('map', 'arm', ours['map']['arm'], comp['map']),
                      ('map', 'cstar', ours['map']['cstar'], ret['aspects'][aspect]['arms'][CSTAR_ARM]['map'])]
            for key, field, a, b in pairs:
                rows.append((arm, aspect, key, field, a, b, abs(a - b) <= TOL_REG))
                worst = max(worst, abs(a - b))
    for r in rows:
        print(f'{r[0]} {r[1]} {r[2]} {r[3]}: ours {r[4]!r} saved {r[5]!r} '
              + ('' if r[3] == 'self' else f'diff {r[4] - r[5]:+.3e} ') + ('ok' if r[6] else 'DIFFERS'))
    ok = all(r[6] for r in rows)
    print(f'regression: {len(rows)} cells, max |diff| {worst:.3e}, {"all equal" if ok else "SOME DIFFER"} (tolerance {TOL_REG})')
    return ok


# ================================================================================================ diagnostics (§3.6)
def silhouette_cosine(x, labels):
    """Mean silhouette of cosine distance 1 - cos on all rows, exact: for a cluster C, the summed distance of row i to C is
    |C| - x_i . S_C (unit rows, S_C the sum of C's rows), so no n x n matrix is needed. A cluster of one row scores 0
    (scikit-learn's convention); `labels` can be any integers. Equals `sklearn.metrics.silhouette_score(metric='cosine')`
    (tested)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    assert x.ndim == 2 and len(labels) == n and np.isfinite(x).all()
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norm > 0).all(), 'a zero row'
    x = x / norm
    ids, inv = np.unique(np.asarray(labels), return_inverse=True)
    k = len(ids)
    assert k >= 2, 'a silhouette needs two clusters'
    size = np.bincount(inv, minlength=k).astype(np.float64)
    sums = np.zeros((k, x.shape[1]))
    for c in range(k):
        sums[c] = x[inv == c].sum(axis=0)
    total = size[None, :] - x @ sums.T                                   # [n, k]: summed distance to each cluster
    own = total[np.arange(n), inv]
    a = own / np.maximum(size[inv] - 1, 1)
    other = total / size[None, :]
    other[np.arange(n), inv] = np.inf
    b = other.min(axis=1)
    denom = np.maximum(a, b)
    s = np.where((size[inv] > 1) & (denom > 0), (b - a) / np.where(denom > 0, denom, 1), 0.0)
    return float(s.mean())


def cluster_shares(labels, k):
    """Largest and smallest cluster's share of all rows over the k clusters (an empty cluster counts as 0)."""
    labels = np.asarray(labels)
    assert labels.min() >= 0 and labels.max() < k, f'labels outside 0..{k - 1}'
    share = np.bincount(labels, minlength=k) / len(labels)
    return dict(largest=float(share.max()), smallest=float(share.min()), n_empty=int((share == 0).sum()),
                n_used=int((share > 0).sum()))


def dec_stats(side):
    """DEC statistics of one head from its training sidecar (`R6-5/<aspect>__<graph>__s<seed>.json`). `history` entries carry
    a `dec` dict when DEC is on: warm-up entries `phase='warmup'` (the last also `start_epoch`), DEC epochs `phase='dec'` with
    `changed`, `away` (fraction away from the DEC-start clusters), `share_max`, `share_min`; the last entry `stop_reason`.
    `dec_epochs` counts `phase == 'dec'` entries only. Stop epoch = the last epoch run (`epochs_run`); the best epoch beside."""
    hist = side['history']
    assert len(hist) == side['epochs_run'] > 0, (len(hist), side['epochs_run'])
    base = dict(best_epoch=int(side['best_epoch']), stop_epoch=int(side['epochs_run']))
    if not any('dec' in h for h in hist):
        return dict(base, dec_on=False, dec_epochs=0, warmup_epochs=0, dec_start_epoch=None, stop_reason=None, away=None,
                    changed=None, share_max=None, share_min=None)
    assert all(isinstance(h.get('dec'), dict) and h['dec'].get('phase') in ('warmup', 'dec') for h in hist), 'a history entry lacks its dec phase'
    phases = [h['dec']['phase'] for h in hist]
    n_warm = phases.count('warmup')
    assert phases == ['warmup'] * n_warm + ['dec'] * (len(hist) - n_warm), 'warm-up and DEC epochs are not in order'
    assert n_warm >= 1, 'no warm-up entry'
    n_dec = len(hist) - n_warm
    start = hist[n_warm - 1]['dec'].get('start_epoch')
    out = dict(base, dec_on=True, dec_epochs=n_dec, warmup_epochs=n_warm, dec_start_epoch=start)
    if n_dec:
        last = hist[-1]['dec']
        assert last.get('stop_reason'), 'the last DEC epoch has no stop reason'
        assert all(h['dec']['start_epoch'] == start for h in hist[n_warm:]), 'DEC start epoch changes'
        assert start == n_warm + 1, (start, n_warm)
        out.update(stop_reason=last['stop_reason'], away=float(last['away']), changed=float(last['changed']),
                   share_max=float(last['share_max']), share_min=float(last['share_min']))
    else:
        out.update(stop_reason=None, away=None, changed=None, share_max=None, share_min=None)
    return out


def record_agrees(stats, head):
    """The run record's per-head fields (`heads.<aspect>`: `stop_reason`, `stop_epoch`, `dec_start_epoch`) against the sidecar's
    statistics: {field: bool}, or None when the record has no such head entry. Shown, not enforced (the record is written by the
    runner, the sidecar by the trainer)."""
    if not head:
        return None
    return dict(stop_reason=head.get('stop_reason') == stats['stop_reason'], stop_epoch=head.get('stop_epoch') == stats['stop_epoch'],
                dec_start_epoch=head.get('dec_start_epoch') == stats['dec_start_epoch'])


def _silhouette_row(x, parts, k):
    """Silhouette against state 0's partition, and the cluster shares of state 0 and the mean over the three states."""
    p0 = parts[0]
    shares = [cluster_shares(p, k) for p in parts]
    return dict(silhouette=silhouette_cosine(x, p0), largest=shares[0]['largest'], smallest=shares[0]['smallest'],
                n_empty=shares[0]['n_empty'], largest_states=[s['largest'] for s in shares],
                smallest_states=[s['smallest'] for s in shares])


def _mean_rows(rows):
    keys = ('silhouette', 'largest', 'smallest', 'n_empty')
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def diag(arms, root, cstar_root, out, workers=8, val_root=None):
    _deps()
    t0 = time.monotonic()
    out, root, cstar_root = Path(out), Path(root), Path(cstar_root)
    check_arm_names(arms, root)
    names, sha, labels = rb_part2_report.held_labels()
    cache = out / 'cache'
    doc = dict(stage='r21_diag', written=time.strftime('%Y-%m-%d %H:%M'), root=str(root), arms=list(arms), seeds=list(SEEDS),
               partition='MiniBatchKMeans at the class count, state 0 (silhouette); shares also over the three states',
               silhouette='cosine, all 12,281 rows, exact (no sampling)', sources={}, dec={}, val={}, **code_state())
    for aspect in ASPECTS:
        k = ns_paths.CLASS_COUNT[(SET, aspect)]
        sources = [(CSTAR_ARM, cstar_root, 'cstar')] + [(a, root, a) for a in arms]
        for arm, r, label in sources:
            rows = {}
            for s in SEEDS:
                x, meta = load_held(r, arm, aspect, s)
                rows[s] = _silhouette_row(x, get_partitions(cache, arm, aspect, s, k, meta, x), k)
            doc['sources'].setdefault(label, {})[aspect] = dict(k=int(k), seeds={str(s): v for s, v in rows.items()},
                                                                 mean=_mean_rows(list(rows.values())))
        for label, x in (('plain', ns_data.plain(SET, 'held', 'qwen')), ('teacher', ns_data.teacher(SET, 'held', aspect))):
            meta = dict(array_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), rows=N_HELD)
            row = _silhouette_row(x, get_partitions(cache, label, aspect, None, k, meta, x), k)
            doc['sources'].setdefault(label, {})[aspect] = dict(k=int(k), seeds={'-': row}, mean=_mean_rows([row]))
        print(f'{aspect}: silhouettes done', flush=True)
    train_root = root / 'train'
    for arm in arms:
        doc['dec'][arm] = {a: {} for a in ASPECTS}
        for a in ASPECTS:
            for s in SEEDS:
                side = SC.load_val(arm, a, s, train_root)[1]
                stats = dec_stats(side)
                rec_path = train_root / arm / f'run__s{s}.json'
                assert 'train' in rec_path.parts and not ({'held', 'test'} & set(rec_path.parts)), rec_path
                head = (json.loads(rec_path.read_text()).get('heads', {}) or {}).get(a) if rec_path.exists() else None
                doc['dec'][arm][a][str(s)] = dict(stats, run_record=head, run_record_agrees=record_agrees(stats, head))
    for arm in [SC.CSTAR] + list(arms):
        doc['val'][arm] = {a: dict(ami=[], map=[]) for a in ASPECTS}
        for s in SEEDS:
            res = SC.val_cached(arm, s, out, None if arm == SC.CSTAR else train_root, workers)
            for a in ASPECTS:
                doc['val'][arm][a]['ami'].append(res['aspects'][a]['ami'])
                doc['val'][arm][a]['map'].append(res['aspects'][a]['map'])
        for a in ASPECTS:
            v = doc['val'][arm][a]
            v['ami_mean'], v['map_mean'] = float(np.mean(v['ami'])), float(np.mean(v['map']))
    doc['seconds'] = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / 'diag.json').write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    (out / 'diag.md').write_text(diag_markdown(doc))
    print(f'{doc["seconds"]:.0f}s -> {out / "diag.json"}, diag.md', flush=True)
    return doc


def diag_markdown(doc):
    L = ['# r21 diagnostics (spec §3.6; they decide nothing)', '',
         'Silhouette: cosine, all 12,281 test paintings against the k-means partition at the class count (state 0); mean over seeds 0 to 4 '
         '(plain Qwen image and teacher: no seeds). Shares: largest / smallest cluster as a share of the test paintings (state 0), mean '
         'over seeds.', '', '## Silhouette and cluster shares (test)', '',
         '| source | ' + ' | '.join(f'{a} silhouette | {a} largest / smallest' for a in ASPECTS) + ' |', '|---|' + '---|---|' * len(ASPECTS)]
    for label, per in doc['sources'].items():
        L.append(f'| {label} | ' + ' | '.join(f"{per[a]['mean']['silhouette']:+.4f} | {per[a]['mean']['largest']:.3f} / {per[a]['mean']['smallest']:.3f}"
                                              for a in ASPECTS) + ' |')
    L += ['', '## DEC epochs, stop and distance from the start clusters (training sidecars, per seed 0 to 4)', '',
          'DEC epochs / stop epoch (the last epoch run) / stop reason / fraction of training paintings away from their DEC-start cluster at the end.', '',
          '| arm | aspect | seeds 0 to 4 |', '|---|---|---|']
    for arm, per in doc['dec'].items():
        for a, seeds in per.items():
            cells = []
            for s, d in seeds.items():
                bad = d['run_record_agrees'] is not None and not all(d['run_record_agrees'].values())
                cells.append((f"{d['dec_epochs']} / {d['stop_epoch']} / {d['stop_reason'] or 'n/a'} / "
                              + ('n/a' if d['away'] is None else f"{d['away']:.3f}")) + (' (run record differs)' if bad else ''))
            L.append(f'| {arm} | {a} | ' + '; '.join(cells) + ' |')
    L += ['', '## Validation AMI (class count) and mAP, seeds 0 to 4 (mean; per seed in the JSON)', '',
          '| arm | ' + ' | '.join(f'{a} AMI / mAP' for a in ASPECTS) + ' |', '|---|' + '---|' * len(ASPECTS)]
    for arm, per in doc['val'].items():
        L.append(f'| {arm} | ' + ' | '.join(f"{per[a]['ami_mean']:.4f} / {per[a]['map_mean']:.4f}" for a in ASPECTS) + ' |')
    return '\n'.join(L) + '\n'


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('table', 'diag'):
        p = sub.add_parser(name)
        p.add_argument('--arms', nargs='+', help="default: dec-off and the stage 1b picks (`--picks`)")
        p.add_argument('--picks', type=Path, default=SCORES / 'stage1_picks.json')
        p.add_argument('--root', type=Path, default=R21_FULL, help="the arms' tree (`held/<arm>/`, `train/<arm>/`)")
        p.add_argument('--cstar-root', type=Path, default=R10_FULL)
        p.add_argument('--out', type=Path, default=SCORES)
        p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
        if name == 'table':
            p.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
            p.add_argument('--ami-draws', type=int, default=N_AMI_DRAWS)
            p.add_argument('--map-draws', type=int, default=N_MAP_DRAWS)
            p.add_argument('--name', default='stage2_table')
    r = sub.add_parser('regress')
    r.add_argument('--table', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'regress':
        sys.exit(0 if regress(args.table) else 1)
    arms = args.arms or picked_arms(args.picks)
    if args.cmd == 'table':
        table(arms, args.root, args.cstar_root, args.out, args.ami_draws, args.map_draws, args.workers, tuple(args.aspects), args.name)
    else:
        diag(arms, args.root, args.cstar_root, args.out, args.workers)


if __name__ == '__main__':
    main()
